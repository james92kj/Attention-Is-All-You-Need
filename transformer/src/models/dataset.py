from torch.utils.data import Dataset
from transformer.src.utils import casual_mask
from tokenizers import Tokenizer
import torch

class BilingualDataset(Dataset):

    def __init__(self, raw_ds: Dataset, src_seq_len: int, tgt_seq_len: int, 
                 src_tokenizer:Tokenizer, tgt_tokenizer:Tokenizer,
                 src_lang: str, tgt_lang: str):
        
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src_seq_len = src_seq_len
        self.tgt_seq_len = tgt_seq_len
        self.raw_ds = raw_ds
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = torch.tensor(
                [self.src_tokenizer.token_to_id('[SOS]')], dtype=torch.int64)
        
        self.eos_token = torch.tensor(
            [self.src_tokenizer.token_to_id('[EOS]')], dtype=torch.int64
        )

        self.pad_token = torch.tensor(
            [self.src_tokenizer.token_to_id('[PAD]')], dtype=torch.int64
        )


    def __len__(self):
        return len(self.raw_ds)

    def __getitem__(self, idx):

        sentence = self.raw_ds[idx]

        source_sentence = sentence['translation'][self.src_lang]
        target_sentence = sentence['translation'][self.tgt_lang]


        # encode the tokens in encoder
        encoder_input_tokens = self.src_tokenizer.encode(source_sentence).ids
        
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(encoder_input_tokens,dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token]*(self.src_seq_len - len(encoder_input_tokens) - 2))
            ],
            dim = 0
        )

        # encode the input in the decode 
        decoder_input_tokens = self.tgt_tokenizer.encode(target_sentence).ids

        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(decoder_input_tokens,dtype=torch.int64),
                torch.tensor([self.pad_token]*(self.tgt_seq_len - len(decoder_input_tokens) - 1))
            ],
            dim = 0
        )

        label = torch.cat(
            [
                torch.tensor(decoder_input_tokens,dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token]*(self.tgt_seq_len - len(decoder_input_tokens) - 1))
            ]
        )

        return {
            'encoder_input': encoder_input,
            'encoder_mask' :  (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            'decoder_input': decoder_input,
            'decoder_mask': (decoder_input != self.pad_token).unsqueeze(0).int() & casual_mask(decoder_input.size()[0]),
            'label': label,
            'src_text': source_sentence,
            'tgt_text': target_sentence
        }

