from datasets import Dataset

def get_ds(dataset:Dataset, lang):
    for record in dataset:
        yield record['translation'][lang]