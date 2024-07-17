import pickle
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# Load the data from the pickle files
with open('data/files.pkl', 'rb') as f:
    files = pickle.load(f)

with open('data/labels.pkl', 'rb') as f:
    labels = pickle.load(f)

with open('data/indices_dict.pkl', 'rb') as f:
    indices_dict = pickle.load(f)

# Prepare DataFrame
df = pd.DataFrame({'text': files, 'label': labels})

# Split the data into train and test sets using indices
train_df = df.iloc[indices_dict['gpt_train'] + indices_dict['human_train']]
test_df = df.iloc[indices_dict['gpt_test'] + indices_dict['human_test']]

# Convert DataFrame to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

model_name = "distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

# Determine the device
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
)

trainer.train()
