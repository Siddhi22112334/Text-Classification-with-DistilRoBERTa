import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

# Load the datasets
full_df = pd.read_csv('data/full.csv')
three_sentence_df = pd.read_csv('data/three_sentence_dataset.csv')
single_paragraph_df = pd.read_csv('data/single_paragraph_dataset.csv')
two_paragraph_df = pd.read_csv('data/two_paragraph_dataset.csv')

# Combine the datasets
combined_df = pd.concat([three_sentence_df, single_paragraph_df, two_paragraph_df, full_df], ignore_index=True)

# Ensure the text column is a string
combined_df['text'] = combined_df['text'].astype(str)

# Split the data into train and test sets
train_df, test_df = train_test_split(combined_df, test_size=0.2, random_state=42)

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

# Remove columns not needed for training to prevent errors
if '__index_level_0__' in tokenized_train_dataset.column_names:
    tokenized_train_dataset = tokenized_train_dataset.remove_columns(['__index_level_0__'])
if '__index_level_0__' in tokenized_test_dataset.column_names:
    tokenized_test_dataset = tokenized_test_dataset.remove_columns(['__index_level_0__'])

# Set the format for PyTorch
tokenized_train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
tokenized_test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

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
