import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from peft import PeftModel
from torch.utils.data import DataLoader
from data_utils import load_and_tokenize_dataset
import evaluate


# Function to classify a single text sample
def classify(model, tokenizer, text, id2label):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    # Tokenize and move input to device
    inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model(**inputs)
    # Get predicted class
    prediction = output.logits.argmax(dim=-1).item()
    print(f'\n Class: {prediction}, Label: {id2label[prediction]}, Text: {text}')
    return id2label[prediction]

# Function to evaluate a model on a given dataset
def evaluate_model(inference_model, dataset, labelled=True, batch_size=8, data_collator=None):
    eval_dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inference_model.to(device)
    inference_model.eval()

    all_predictions = []
    if labelled:
        # Load accuracy metric if dataset has labels
        metric = evaluate.load('accuracy')

    # Iterate through batches and make predictions
    for batch in tqdm(eval_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = inference_model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        all_predictions.append(predictions.cpu())

        # Update accuracy metric if labels available
        if labelled:
            metric.add_batch(predictions=predictions.cpu().numpy(), references=batch["labels"].cpu().numpy())

    all_predictions = torch.cat(all_predictions, dim=0)
    if labelled:
        eval_metric = metric.compute()
        print("Evaluation Metric:", eval_metric)
        return eval_metric, all_predictions
    return all_predictions


# To load model, prepare data, and run inference
def run_inference():
    model_path = "results/r4_a16_d0_blora_only_oadamw_torch_lr0.001_bs16_ep1"
    num_labels = 4
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    base_model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=num_labels)

    # Load fine-tuned PEFT model
    peft_model = PeftModel.from_pretrained(base_model, model_path)
    peft_model.eval()
    peft_model.print_trainable_parameters()

    # Load and prepare evaluation dataset
    dataset, tokenized_dataset, _ = load_and_tokenize_dataset()
    eval_dataset = tokenized_dataset.train_test_split(test_size=640, seed=42)['test']
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    # Evaluate model on labelled data
    _, _ = evaluate_model(peft_model, eval_dataset, True, 8, data_collator)

    # Load unlabelled test dataset
    unlabelled_dataset = pd.read_pickle("test_unlabelled.pkl")

    # Tokenize unlabelled dataset
    def preprocess(examples):
        return tokenizer(examples['text'], truncation=True, padding=True)
    test_dataset = unlabelled_dataset.map(preprocess, batched=True, remove_columns=["text"])

    # Run inference on unlabelled data
    preds = evaluate_model(peft_model, test_dataset, False, 8, data_collator)
    # Save predictions to CSV
    df_output = pd.DataFrame({
        'ID': range(len(preds)),
        'Label': preds.numpy()
    })
    df_output.to_csv(os.path.join("results", "inference_output.csv"), index=False)
    print("Inference complete. Predictions saved to inference_output.csv")
