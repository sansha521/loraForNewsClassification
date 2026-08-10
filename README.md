# News Classification with LoRA (RoBERTa)

This repository contains code and notebooks for fine-tuning RoBERTa using Low-Rank Adaptation (LoRA) for the AG News text classification task. The project was developed for a Kaggle-style experiment with the constraint that the final model should use no more than 1 million trainable parameters.

Key results
- Best test accuracy: 84.3%
- Best hyperparameters: r = 4, alpha = 16, dropout = 0, bias = "lora_only"
- Training details: Adam optimizer, batch size 16, learning rate 1e-3

Background
- LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning method that injects trainable low-rank matrices into the transformer architecture instead of updating the full pre-trained weight matrices. This allows large models to be adapted to downstream tasks while keeping the number of trainable parameters small.
- Paper: https://arxiv.org/abs/2106.09685
- Implementation: Hugging Face PEFT (Parameter-Efficient Fine-Tuning)

Repository structure
- notebooks/
  - LORA_for_news_classification.ipynb  — main notebook used for training, evaluation, and inference
- data/
  - (expected) train/val/test datasets or scripts to prepare AG News
- results/
  - inference_output.csv — model inferences on the unlabelled test set
- checkpoints/
  - checkpoint-*/ — saved model checkpoints and model card templates
- requirements.txt — Python package dependencies

Quick start
1. Create a Python environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Prepare the AG News dataset (or use the provided/preprocessed files):
- The notebook expects training/validation/test splits. If you have raw AG News data, add a short preprocessing script to `data/` or adapt the notebook.

3. Open the main notebook and run the cells:
- `notebooks/LORA_for_news_classification.ipynb` contains step-by-step code to reproduce training, evaluation, and inference.

Reproducing experiments (summary)
- Tested configurations (examples):
  - r = [4, 8, 21]
  - alpha = [16, 24, 32]
  - dropout = [0, 0.05, 0.1]
  - bias = ["lora_only"]
  - optimizer = [adamw_torch, muon]
  - learning_rate = [1e-3, 1e-2, 1e-1]
  - epochs = [1]

- The best run achieved 84.3% test accuracy with r=4, alpha=16, dropout=0, and "lora_only" bias. Training used AdamW, lr=1e-3, batch_size=16.

Training notes
- Experiments were run on an HPC burst node (4 GPUs, 12 CPU cores). If you do not have multiple GPUs available, reduce batch size or run on a single GPU.
- The project is implemented using PyTorch and Hugging Face Transformers + PEFT. See `requirements.txt` for exact versions.

Inference
- The `results/inference_output.csv` file contains model predictions on the unlabelled test set (`test_unlabelled.pkl` in the repo, if present). The notebook includes the inference pipeline used to generate this file.

Model checkpoints and model card
- Checkpoints are stored under `checkpoints/`. Some checkpoints include a model card template (incomplete) that can be filled with additional metadata.

References
- LoRA paper: https://arxiv.org/abs/2106.09685
- Hugging Face PEFT: https://github.com/huggingface/peft

Contributing
- Contributions are welcome. Open an issue or submit a PR for bug fixes, improvements, or updated experiments.

License

This repository is licensed under the MIT License — see the LICENSE file for details.

Unless otherwise noted, the MIT license applies to the code, notebooks, and model artifacts (checkpoints and inference outputs) included in this repository. The original AG News dataset is not included in this repository and remains subject to its original license and terms of use.

Contact
- Maintainer: sansha521

