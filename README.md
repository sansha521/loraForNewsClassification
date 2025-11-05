# Team YAN: Deep Learning CS 6953 / ECE 7123 2025 Spring

This is the project respository for team YAN - Nidhi, Athul, Yumiko

## Install Libraries
Install the libraries with the specific version mentioned in requirements.txt

## Training.
We had trained the models in the HPC burst node for efficient compute (4 GPUs and 12 cores).

## Final Model:
The final model is at `notebooks/LORA_for_news_classification.ipynb`

results folder contains the `inference_output.csv` which are the inferences made by the model on unlabelled data (`test_unlabelled.pkl`) 

This is a project repo for a Kaggle competition with the goal of coming up with a modified BERT (RoBERTa) architecture with the highest test accuracy for text classification on a dataset "AGNEWS". 
The final model must follow the constraint that it uses no more than 1 million trainable parameters. 

The team worked on modifying a specific part of BERT - the low rank adaption (LoRA). 

LoRA is a finetuning method where instead of finetuning all the weights that constitute the weight matrix of the pre-trained large language model, two smaller matrices that approximate this larger matrix are fine-tuned (see databricks article: https://www.databricks.com/blog/efficient-fine-tuning-lora-guide-llms).
It is implemented in the Hugging Face Parameter Efficient Fine-Tuning (PEFT) library. 
The low-rank matrix derived by LoRA is then feeded to BERT's weight matrices. 

An overview of the LoRA architecture is shown:

<img width="737" height="619" alt="image" src="https://github.com/user-attachments/assets/7ed1cda4-5f5e-4a28-804c-7ebddf84f2c3" />

Source: https://arxiv.org/abs/2106.09685

This represents the tensor operations for one matrix in the model. A and B are the small matrices. 
The input vector d is processed both through the original pre-trained weights and through LoRA's fine-tuned, low-rank decomposition matrices in parallel (see more here: https://www.anyscale.com/blog/fine-tuning-llms-lora-or-full-parameter-an-in-depth-analysis-with-llama-2)

The parameters to tune for this task is the rank of the matrix AB, denoted by r. 
Another parameter is alpha, which is the scalar multiplied with AB when added to the original weight. 

The task allows tuning the parameters by layer or even by weight matrix. 

Additionally, the following can be tuned:
• any optimizer (ADAM, RMSProp, Muon, etc)
• any data filtering strategy (e.g. you can throw away lengthy or weird reviews
according to metrics of your choice.)
• any regularizer
• any choice of learning rate, batch size, epochs, scheduler, etc.
• other tricks such as teacher-student distillation, or quantization (QLoRA), or etc.
• any data augmentation strategy (e.g. you can choose to reword/rephrase/cleanup
training samples using whatever technique you like). 

NYU HPC is used to train the models. 


