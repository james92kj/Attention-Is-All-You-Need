from transformer.src.models import BilingualDataset
from .data_utils import get_max_len
from datasets import Dataset, load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from torch.utils.data import random_split, DataLoader
from pathlib import Path
import os

def get_records(dataset:Dataset, lang):
    for record in dataset:
        yield record['translation'][lang]


def build_tokenizer(cfg , ds, lang):
    tokenizer_path = os.path.join(cfg.artifacts, cfg.train.tokenizer_path.format(lang))
    if not Path(tokenizer_path).exists():
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()

        trainer = WordLevelTrainer(special_tokens=['[SOS]','[EOS]','[UNK]','[PAD]'], min_frequency=2)
        tokenizer.train_from_iterator(get_records(ds, lang),trainer=trainer)
        # TODO @James Need to be fixed
        tokenizer.save(tokenizer_path)
    else:
        tokenizer = Tokenizer.from_file(tokenizer_path)
    
    return tokenizer


def get_ds(cfg):
    # load the dataset
    ds = load_dataset(cfg.dataset.name, f'{cfg.dataset.src_lang}-{cfg.dataset.tgt_lang}', split='train')

    # build the tokenizer for source language 
    src_tokenizer = build_tokenizer(cfg, ds, 'en')
    tgt_tokenizer = build_tokenizer(cfg, ds, 'it')

    src_seq_len = get_max_len(ds,src_tokenizer,cfg.dataset.src_lang)
    tgt_seq_len = get_max_len(ds,tgt_tokenizer,cfg.dataset.tgt_lang)

    # split the dataset 
    train_ds_size = int(0.9 * len(ds))
    val_ds_size = len(ds) - train_ds_size
    train_ds, valid_ds = random_split(ds, [train_ds_size, val_ds_size])

    # Plug the BilingualDataset 
    train_ds = BilingualDataset(train_ds,
                    src_seq_len=src_seq_len,
                    tgt_seq_len=tgt_seq_len,
                    src_tokenizer=src_tokenizer,
                    tgt_tokenizer=tgt_tokenizer,
                    src_lang=cfg.dataset.src_lang,
                    tgt_lang=cfg.dataset.tgt_lang
                    )
    
    valid_ds = BilingualDataset(valid_ds,
                    src_seq_len=src_seq_len,
                    tgt_seq_len=tgt_seq_len,
                    src_tokenizer=src_tokenizer,
                    tgt_tokenizer=tgt_tokenizer,
                    src_lang=cfg.dataset.src_lang,
                    tgt_lang=cfg.dataset.tgt_lang
                    )

    train_dl = DataLoader(train_ds, batch_size=cfg.train.per_device_train_batch_size, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=cfg.train.per_device_valid_batch_size, shuffle=True)


    return {
            'train_dl':train_dl,
            'valid_dl':valid_dl,
            'src_tokenizer':src_tokenizer,
            'tgt_tokenizer':tgt_tokenizer,
            'src_seq_len': src_seq_len,
            'tgt_seq_len': tgt_seq_len
        }


