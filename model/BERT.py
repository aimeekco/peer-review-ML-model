import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import torch
import numpy as np

# read data
processed_data_path = "/Users/aimeeco/peer-review-ML-model/data/processed_data.csv"
df = pd.read_csv(processed_data_path)

texts = df['normalized_sentence'].tolist()
print(f"Number of texts before filtering: {len(texts)}")

texts = [text for text in texts if isinstance(text, str)]
print(f"Number of texts after filtering: {len(texts)}")
print(f"Sample texts: {texts[:5]}")

# filter the original df to match the filtered texts
df_filtered = df[df['normalized_sentence'].apply(lambda x: isinstance(x, str))]

# convert labels to strings and then to lists
def convert_labels_to_list(label):
    if isinstance(label, float):
        return []
    label_string = str(label).strip("[]").replace("'", "")
    return [lbl.strip() for lbl in label_string.split(',')]

df_filtered['section_label'] = df_filtered['section_label'].apply(convert_labels_to_list)
df_filtered['aspect_label'] = df_filtered['aspect_label'].apply(convert_labels_to_list)
df_filtered['purpose_label'] = df_filtered['purpose_label'].apply(convert_labels_to_list)
df_filtered['significance_label'] = df_filtered['significance_label'].apply(convert_labels_to_list)

labels = df_filtered[['section_label', 'aspect_label', 'purpose_label', 'significance_label']].apply(lambda row: row[0] + row[1] + row[2] + row[3], axis=1).tolist()

# convert labels to multi-hot encoded format
mlb = MultiLabelBinarizer()
encoded_labels = mlb.fit_transform(labels)

assert len(texts) == len(encoded_labels), f"The number of texts ({len(texts)}) and labels ({len(encoded_labels)}) should match"

# tokenize data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)

# check device availability
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

class PaperDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]).to(device) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float).to(device)
        return item
    
    def __len__(self):
        return len(self.labels)

dataset = PaperDataset(encodings, encoded_labels)
train_dataset, val_dataset = train_test_split(dataset, test_size=0.2)

# model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(mlb.classes_))
model.to(device)

# calculate class weights manually
label_counts = np.sum(encoded_labels, axis=0)
class_weights = len(encoded_labels) / (len(mlb.classes_) * label_counts)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

print(f"Class Weights: {class_weights}")

# loss function with class weights and custom trainer
from torch.nn import BCEWithLogitsLoss

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels").to(device)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = BCEWithLogitsLoss(pos_weight=model.class_weights.to(device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

model.class_weights = class_weights

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    save_strategy="epoch",
)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()

model.save_pretrained('./results')
tokenizer.save_pretrained('./results')

