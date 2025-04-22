import pandas as pd
from transformers import RobertaTokenizer
from datasets import load_dataset

def load_and_tokenize_dataset(base_model='roberta-base'):
    dataset = load_dataset('ag_news', split='train')
    tokenizer = RobertaTokenizer.from_pretrained(base_model)

    def preprocess(examples):
        return tokenizer(examples['text'], truncation=True, padding=True)

    tokenized_dataset = dataset.map(preprocess, batched=True, remove_columns=["text"])
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")

    return dataset, tokenized_dataset, tokenizer

