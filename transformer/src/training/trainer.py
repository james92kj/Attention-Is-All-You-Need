from transformer import CONF_DIR_PATH, ARTIFACTS_DIR_PATH
from transformer.src.models import build_transformer
from transformer.src.training import run_validation
from transformer.src.utils import get_ds
from tqdm.auto import tqdm
import torch.optim as optim
import torch
import hydra
import os


@hydra.main(config_path=CONF_DIR_PATH, config_name='cfg', version_base=None)
def train(cfg):
    os.makedirs(ARTIFACTS_DIR_PATH,exist_ok=True)

    ds = get_ds(cfg,quick_test=False)
    src_tokenizer, tgt_tokenizer = ds['src_tokenizer'], ds['tgt_tokenizer']
    train_dl, valid_dl = ds['train_dl'],ds['valid_dl']
    src_seq_len, tgt_seq_len = ds['src_seq_len'], ds['tgt_seq_len']

    print(f'The length of the records in train is {len(train_dl)}')
    print(f'The length of the records in valid is {len(valid_dl)}')

    # get the vocab size 
    src_vocab_size = src_tokenizer.get_vocab_size()
    tgt_vocab_size = tgt_tokenizer.get_vocab_size()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
     # get the model 
    model = build_transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        src_seq_len=cfg.model.seq_len,
        tgt_seq_len=cfg.model.seq_len,
        d_model=cfg.model.d_model,
        N=cfg.model.num_layers,
        h=cfg.model.num_heads,
        d_ff=cfg.model.d_ff,
        dropout=cfg.model.dropout
    )
    
    model.to(device=device)
    # get the optimizer 
    optimizer = optim.Adam(model.parameters(),lr=cfg.train.lr, eps=1e-9)

    # setup the train function
    initial_epoch = 0 
    global_step = 0

    end_epoch = cfg.train.num_epochs

    loss_fn = torch.nn.CrossEntropyLoss(
                ignore_index=src_tokenizer.token_to_id('[PAD]'), 
                label_smoothing=0.1
            ).to(device=device)
    

    for epoch in range(initial_epoch, end_epoch):
    
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dl, desc=f'Epoch {epoch}')
        
        for step, batch in enumerate(batch_iterator):
            
            label = batch['label'].to(device = device)
            encoder_input = batch['encoder_input'].to(device=device)
            encoder_mask = batch['encoder_mask'].to(device=device)

            decoder_input = batch['decoder_input'].to(device=device)
            decoder_mask = batch['decoder_mask'].to(device=device)

            encoder_output = model.encode(encoder_input,src_mask=encoder_mask)
            decoder_output = model.decode(encoder_output=encoder_output,
                                          tgt_mask=decoder_mask,
                                          src_mask=encoder_mask,
                                          tgt=decoder_input
                                          )
            
            predicted_vocab = model.project(decoder_output)
            loss = loss_fn(predicted_vocab.contiguous().reshape(-1, tgt_vocab_size), label.view(-1))
            loss.backward()

            optimizer.step()
            optimizer.zero_grad() 

            global_step += 1

            if step % 300 == 0:
                run_validation(valid_dl, model,
                    device, tgt_tokenizer,max_len=cfg.model.seq_len, num_examples=2)




    # setup the validation function 

if __name__ =='__main__':
    train()
