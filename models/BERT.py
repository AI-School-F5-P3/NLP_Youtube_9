import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import os

import torch
from torch.utils.data import Dataset

current_dir = os.getcwd()
data_dir = os.path.join(current_dir, '..', 'data')
file_path = os.path.join(data_dir, 'preprocessed_data_novector.csv')

df = pd.read_csv(file_path)

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

train_texts = train_df['Text'].tolist()
train_labels = train_df['IsHateSpeech'].tolist()

val_texts = val_df['Text'].tolist()
val_labels = val_df['IsHateSpeech'].tolist()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the training data
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)

# Tokenize the validation data
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

class HateSpeechDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Create the dataset objects
train_dataset = HateSpeechDataset(train_encodings, train_labels)
val_dataset = HateSpeechDataset(val_encodings, val_labels)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    evaluation_strategy="epoch",
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    return {'accuracy': acc, 'f1': f1}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

save_path = os.path.join(current_dir, 'fine_tuned_hate_speech_model')
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)