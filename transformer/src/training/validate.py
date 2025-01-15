from transformer.src.models import Transformer, greedy_decode
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
import torch 

def run_validation(valid_dl: DataLoader, model:Transformer,
                device:str, tgt_tokenizer: Tokenizer, max_len:int, num_examples=100, ):

    model.eval()
    count  = 0
    source, target, predicted = [], [], []

    with torch.no_grad():

        for batch in valid_dl:
            count += 1
            # extract the input 
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            
            assert encoder_input.size()[0] == 1, 'Please provide a valid batch size'

            tokens = greedy_decode(source=encoder_input,
                source_mask=encoder_mask,
                model=model,
                device=device,
                dst_tokenizer=tgt_tokenizer, max_len=max_len)

            source_txt = batch['src_text']
            target_txt = batch['tgt_text']

            predicted_text = tgt_tokenizer.decode(tokens.detach().cpu().numpy())
    
            source.append(source_txt)
            target.append(target_txt)
            predicted.append(predicted_text)

            print(f"{f'SOURCE: ':>12}{source_txt}")
            print(f"{f'TARGET: ':>12}{target_txt}")
            print(f"{f'PREDICTED: ':>12}{predicted_text}")

            if count == num_examples:
                break
            
            
        

        



