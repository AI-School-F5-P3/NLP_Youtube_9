import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, TrainerCallback, EarlyStoppingCallback
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.nn import CrossEntropyLoss
import os
import mlflow
import mlflow.pytorch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class MetricsCallback(TrainerCallback):
    def __init__(self):
        self.metrics = {'accuracy': [], 'f1': [], 'epoch': []}
        
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            self.metrics['accuracy'].append(metrics.get('eval_accuracy'))
            self.metrics['f1'].append(metrics.get('eval_f1'))
            self.metrics['epoch'].append(state.epoch)
    
    def plot_metrics(self):
        plt.figure(figsize=(10, 5))
        # Accuracy plot
        plt.subplot(1, 2, 1)
        plt.plot(self.metrics['epoch'], self.metrics['accuracy'], label='Accuracy', marker='o')
        plt.title('Accuracy Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.legend()
        
        # F1 Score plot
        plt.subplot(1, 2, 2)
        plt.plot(self.metrics['epoch'], self.metrics['f1'], label='F1 Score', marker='o', color='orange')
        plt.title('F1 Score Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

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

# Tokenize the training and validation data
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

# Dataset class
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

# Create dataset objects
train_dataset = HateSpeechDataset(train_encodings, train_labels)
val_dataset = HateSpeechDataset(val_encodings, val_labels)

# Define class weights and convert to tensor
class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=train_labels)
class_weights = torch.tensor(class_weights, dtype=torch.float)

# Custom model with class-weighted loss
class WeightedBertForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config, class_weights):
        super().__init__(config)
        self.class_weights = class_weights

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
        logits = outputs.logits
        if labels is not None:
            # Apply class weights to loss
            loss_fct = CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            outputs = (loss,) + outputs[1:]
        return outputs

# Instantiate the model with class weights
model = WeightedBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2, class_weights=class_weights)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=6,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=3e-5,
    warmup_steps=500,
    weight_decay=0.02,
    logging_dir='./logs',
    eval_strategy="epoch",  # Evaluate at the end of each epoch
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Define metric computation
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    return {'accuracy': acc, 'f1': f1}

metrics_callback = MetricsCallback()

# Initialize trainer with early stopping callback
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# Train the model
trainer.train()

# Save model and tokenizer
save_path = os.path.join(current_dir, 'fine_tuned_hate_speech_model_3')
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

# Log metrics to MLflow
eval_result = trainer.evaluate()
mlflow.log_metric('accuracy', eval_result['eval_accuracy'])
mlflow.log_metric('f1_score', eval_result['eval_f1'])

# Log model to MLflow
mlflow.pytorch.log_model(model, "model")

# End the MLflow run
mlflow.end_run()