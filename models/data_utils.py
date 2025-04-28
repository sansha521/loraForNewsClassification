import pandas as pd
from transformers import RobertaTokenizer
from datasets import load_dataset

def load_and_tokenize_dataset(base_model='roberta-base'):
    # Load the AG News dataset's training split
    dataset = load_dataset('ag_news', split='train')
    # Initialize a tokenizer from the specified base model
    tokenizer = RobertaTokenizer.from_pretrained(base_model)

    # Define a preprocessing function to tokenize the text

    def preprocess(examples):
        return tokenizer(examples['text'], truncation=True, padding=True) # Tokenize the 'text' field, enabling truncation and padding

    tokenized_dataset = dataset.map(preprocess, batched=True, remove_columns=["text"])
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")

    # Return the original dataset, the tokenized dataset, and the tokenizer
    return dataset, tokenized_dataset, tokenizer

