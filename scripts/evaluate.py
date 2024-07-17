from transformers import Trainer
import pickle
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments

# Load the data from the pickle files
with open('data/files.pkl', 'rb') as f:
    files = pickle.load(f)

with open('data/labels.pkl', 'rb') as f:
    labels = pickle.load(f)

with open('data/indices_dict.pkl', 'rb') as f:
    indices_dict = pickle.load(f)

# Prepare DataFrame
df = pd.DataFrame({'text': files, 'label': labels})

# Split the data into test sets using indices
test_df = df.iloc[indices_dict['gpt_test'] + indices_dict['human_test']]

# Convert DataFrame to Hugging Face Dataset
test_dataset = Dataset.from_pandas(test_df)

model_name = "distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

# Determine the device
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)

training_args = TrainingArguments(
    per_device_eval_batch_size=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=tokenized_test_dataset,
)

results = trainer.evaluate()
print(results)
