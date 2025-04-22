from transformers import RobertaForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from peft import LoraConfig, get_peft_model
from itertools import product
from data_utils import load_and_tokenize_dataset
from metrics import compute_metrics


def run_training():
    dataset, tokenized_dataset, tokenizer = load_and_tokenize_dataset()
    num_labels = dataset.features['label'].num_classes
    class_names = dataset.features["label"].names
    id2label = {i: label for i, label in enumerate(class_names)}

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    split_datasets = tokenized_dataset.train_test_split(test_size=640, seed=42)
    train_dataset = split_datasets['train']
    eval_dataset = split_datasets['test']

    # Grid search config
    lora_rs = [4, 8]
    lora_alphas = [16, 24, 32]
    lora_dropouts = [0, 0.05, 0.1]
    bias_options = ["lora_only"]
    optimizers = ["adamw_torch", 'muon']
    learning_rates = [1e-3, 1e-2, 1e-1]
    batch_sizes = [16, 32]
    epochs = [10]

    results = []
    output_dir = "results"

    for (r, alpha, dropout, bias, opt, lr, bs, ep) in product(
            lora_rs, lora_alphas, lora_dropouts, bias_options, optimizers, learning_rates, batch_sizes, epochs):

        print(f"\n🔧 Training with r={r}, alpha={alpha}, dropout={dropout}, bias={bias}, opt={opt}, lr={lr}, bs={bs}, epochs={ep}")
        model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=num_labels, id2label=id2label)
        peft_config = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, bias=bias, target_modules=["query"], task_type="SEQ_CLS")
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        output_subdir = f"{output_dir}/r{r}_a{alpha}_d{dropout}_b{bias}_o{opt}_lr{lr}_bs{bs}_ep{ep}"

        training_args = TrainingArguments(
            output_dir=output_subdir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
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

        trainer.train()
        metrics = trainer.evaluate()
        model.save_pretrained(output_subdir)
        tokenizer.save_pretrained(output_subdir)

        results.append({
            "r": r, "alpha": alpha, "dropout": dropout, "bias": bias, "optimizer": opt,
            "lr": lr, "batch_size": bs, "epochs": ep, "accuracy": metrics["eval_accuracy"]
        })
