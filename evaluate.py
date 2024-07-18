import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load the test dataset
test_df = pd.read_csv('data/test_dataset.csv')  # Assuming the test dataset is saved as 'data/test_dataset.csv'

# Ensure the text column is a string
test_df['text'] = test_df['text'].astype(str)

# Convert DataFrame to Hugging Face Dataset
test_dataset = Dataset.from_pandas(test_df)

model_name = "distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

# Remove columns not needed for evaluation
if '__index_level_0__' in tokenized_test_dataset.column_names:
    tokenized_test_dataset = tokenized_test_dataset.remove_columns(['__index_level_0__'])

# Set the format for PyTorch
tokenized_test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Determine the device
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)

# Define evaluation function
def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

# Set training arguments
training_args = TrainingArguments(
    per_device_eval_batch_size=2,
    output_dir='./results',
    logging_dir='./logs',
    do_train=False,
    do_eval=True,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=tokenized_test_dataset,
    compute_metrics=compute_metrics,
)

# Evaluate the model
results = trainer.evaluate()
print(results)
