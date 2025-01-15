from tokenizers import Tokenizer
from torch.utils.data import Dataset
import torch

def get_max_len(dl: Dataset, tokenizer:Tokenizer, lang):
    max_len = 0
    for record in dl:
        sentence = record['translation'][lang]
        max_len = max(max_len, len(tokenizer.encode(sentence).ids))

    return max_len


def casual_mask(size: int):
    mat = torch.triu(torch.ones(1,size,size), diagonal=1).int()
    return mat == 0