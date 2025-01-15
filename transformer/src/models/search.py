from transformer.src.utils import casual_mask
from transformer.src.models import Transformer
from tokenizers import Tokenizer
import torch 

def greedy_decode(source: torch.Tensor, source_mask: torch.Tensor, 
                  model:Transformer, device:str, 
                  dst_tokenizer: Tokenizer, max_len: int):

    # set the model in eval model 
    model.eval()

    sos_idx = torch.tensor(dst_tokenizer.token_to_id ('[SOS]')).type_as(source)
    end_idx = torch.tensor(dst_tokenizer.token_to_id('[EOS]')).type_as(source)

    encoder_input = source.to(device=device)
    encoder_mask  = source_mask.to(device=device)

    decoder_input = torch.empty((1,1)).fill_(sos_idx).type_as(encoder_input).to(device=device)
    decoder_mask = casual_mask(decoder_input.size()[1]).type_as(encoder_mask).to(device=device)

    encoder_output = model.encode(encoder_input, encoder_mask)
    
    while True:

        if decoder_input.size(1) == max_len:
            break
        
        decoder_output = model.decode(
            encoder_output=encoder_output,tgt_mask=decoder_mask,
            src_mask=encoder_mask,tgt=decoder_input
            )
        last_token = decoder_output[:,-1]
        _, next_token = torch.max(last_token, dim=-1)
        
        decoder_input = torch.cat([
                                decoder_input, 
                                torch.empty((1,1)).fill_(next_token.item())
                                    .type_as(encoder_input)
                                    .to(device=device)
                                ], dim=-1)
        
        decoder_mask = casual_mask(decoder_input.size()[1]).type_as(encoder_mask).to(device=device)

        if next_token == end_idx:
            break

    return decoder_input.squeeze(0)