from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import torch
import pandas as pd

processed_data_path = "/Users/aimeeco/peer-review-ML-model/data/processed_data.csv"
df = pd.read_csv(processed_data_path)
texts = df['normalized_sentence'].tolist()
labels = df['tags'].tolist()

# tokenize 
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
encodings = tokenizer(texts, truncation=True, padding=True)

# prep dataset
class PaperDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)
    
dataset = PaperDataset(encodings, labels)
train_dataset, val_dataset = train_test_split(dataset, test_size=0.2)

# load model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

#training arguements 
training_args = TrainingArguments(
    output_dir = './results',
    num_train_epochs = 3,
    per_device_train_batch_size = 8,
    per_device_eval_batch_size = 8,
    warmup_steps = 500,
    weight_decay = 0.01,
    logging_dir = './logs',
)

#trainer 
trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = train_dataset,
    eval_dataset = val_dataset
)

trainer.train()