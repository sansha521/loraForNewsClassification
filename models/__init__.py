import os
import pandas as pd
import torch
from transformers import RobertaModel, RobertaTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding, RobertaForSequenceClassification
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import load_dataset, Dataset, ClassLabel
import pickle
from itertools import product
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader
import evaluate
from tqdm import tqdm


base_model = 'roberta-base'

dataset = load_dataset('ag_news', split='train')
tokenizer = RobertaTokenizer.from_pretrained(base_model)

def preprocess(examples):
    tokenized = tokenizer(examples['text'], truncation=True, padding=True)
    return tokenized

tokenized_dataset = dataset.map(preprocess, batched=True,  remove_columns=["text"])
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")


# Extract the number of classess and their names
num_labels = dataset.features['label'].num_classes
class_names = dataset.features["label"].names
print(f"number of labels: {num_labels}")
print(f"the labels: {class_names}")

# Create an id2label mapping
# We will need this for our classifier.
id2label = {i: label for i, label in enumerate(class_names)}

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

# Loading pre-trained model
model = RobertaForSequenceClassification.from_pretrained(
    base_model,
    id2label=id2label)

# Split the original training set
split_datasets = tokenized_dataset.train_test_split(test_size=640, seed=42)
train_dataset = split_datasets['train']
eval_dataset = split_datasets['test']

# configs for grid search
lora_rs = [4, 8]
lora_alphas = [16, 24, 32]
lora_dropouts = [0, 0.05, 0.1]
bias_options = ["lora_only"]
optimizers = ["adamw_torch", 'muon']
learning_rates = [1e-3, 1e-2, 1e-1]
batch_sizes = [16, 32]
epochs = [10]

# To track evaluation accuracy during training
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # Calculate accuracy
    accuracy = accuracy_score(labels, preds)
    return {
        'accuracy': accuracy
    }


results = []
output_dir = "results"

for (r, alpha, dropout, bias, opt, lr, bs, ep) in product(
        lora_rs, lora_alphas, lora_dropouts, bias_options, optimizers, learning_rates, batch_sizes, epochs):

    print(f"\n🔧 Training with r={r}, alpha={alpha}, dropout={dropout}, bias={bias}, opt={opt}, lr={lr}, bs={bs}, epochs={ep}")

    # Fresh base model
    base_model_path = 'roberta-base'
    model = RobertaForSequenceClassification.from_pretrained(
        base_model_path, num_labels=num_labels, id2label=id2label)

    # LoRA config
    peft_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias=bias,
        target_modules=["query"],
        task_type="SEQ_CLS"
    )

    model = get_peft_model(model, peft_config)

    # Print trainable params
    model.print_trainable_parameters()

    # Output directory for saving model + results
    output_subdir = f"{output_dir}/r{r}_a{alpha}_d{dropout}_b{bias}_o{opt}_lr{lr}_bs{bs}_ep{ep}"

    training_args = TrainingArguments(
        output_dir=output_subdir,
        evaluation_strategy="epoch",
        save_strategy="epoch",              # ensure saving
        save_total_limit=1,                 # optional: save last only
        learning_rate=lr,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=64,
        num_train_epochs=ep,
        optim=opt,
        report_to=None,
        logging_steps=100,
        disable_tqdm=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    # Train and evaluate
    trainer.train()
    metrics = trainer.evaluate()

    # Save PEFT model (saves adapter config + weights)
    model.save_pretrained(output_subdir)

    # Optionally save tokenizer (useful for later inference)
    tokenizer.save_pretrained(output_subdir)

    # Record result
    results.append({
        "r": r,
        "alpha": alpha,
        "dropout": dropout,
        "bias": bias,
        "optimizer": opt,
        "lr": lr,
        "batch_size": bs,
        "epochs": ep,
        "accuracy": metrics["eval_accuracy"]
    })


model_path = "results/r4_a16_d0_blora_only_oadamw_torch_lr0.001_bs16_ep1"

num_labels = 4
tokenizer = AutoTokenizer.from_pretrained(model_path)
base_model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels = num_labels)


peft_model = PeftModel.from_pretrained(base_model, model_path)
peft_model.eval()

peft_model.print_trainable_parameters()

def classify(model, tokenizer, text):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model(**inputs)

    prediction = output.logits.argmax(dim=-1).item()

    print(f'\n Class: {prediction}, Label: {id2label[prediction]}, Text: {text}')
    return id2label[prediction]

classify( peft_model, tokenizer, "Kederis proclaims innocence Olympic champion Kostas Kederis today left hospital ahead of his date with IOC inquisitors claiming his ...")
classify( peft_model, tokenizer, "Wall St. Bears Claw Back Into the Black (Reuters) Reuters - Short-sellers, Wall Street's dwindling\band of ultra-cynics, are seeing green again.")

def evaluate_model(inference_model, dataset, labelled=True, batch_size=8, data_collator=None):
    """
    Evaluate a PEFT model on a dataset.

    Args:
        inference_model: The model to evaluate.
        dataset: The dataset (Hugging Face Dataset) to run inference on.
        labelled (bool): If True, the dataset includes labels and metrics will be computed.
                         If False, only predictions will be returned.
        batch_size (int): Batch size for inference.
        data_collator: Function to collate batches. If None, the default collate_fn is used.

    Returns:
        If labelled is True, returns a tuple (metrics, predictions)
        If labelled is False, returns the predictions.
    """
    # Create the DataLoader
    eval_dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inference_model.to(device)
    inference_model.eval()

    all_predictions = []
    if labelled:
        metric = evaluate.load('accuracy')

    # Loop over the DataLoader
    for batch in tqdm(eval_dataloader):
        # Move each tensor in the batch to the device
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = inference_model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        all_predictions.append(predictions.cpu())

        if labelled:
            # Expecting that labels are provided under the "labels" key.
            references = batch["labels"]
            metric.add_batch(
                predictions=predictions.cpu().numpy(),
                references=references.cpu().numpy()
            )

    # Concatenate predictions from all batches
    all_predictions = torch.cat(all_predictions, dim=0)

    if labelled:
        eval_metric = metric.compute()
        print("Evaluation Metric:", eval_metric)
        return eval_metric, all_predictions
    else:
        return all_predictions
    

# Check evaluation accuracy
_, _ = evaluate_model(peft_model, eval_dataset, True, 8, data_collator)

#Load your unlabelled data
unlabelled_dataset = pd.read_pickle("test_unlabelled.pkl")
test_dataset = unlabelled_dataset.map(preprocess, batched=True, remove_columns=["text"])




# Run inference and save predictions
preds = evaluate_model(peft_model, test_dataset, False, 8, data_collator)
df_output = pd.DataFrame({
    'ID': range(len(preds)),
    'Label': preds.numpy()  # or preds.tolist()
})
df_output.to_csv(os.path.join(output_dir,"inference_output.csv"), index=False)
print("Inference complete. Predictions saved to inference_output.csv")