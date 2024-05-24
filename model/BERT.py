import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import torch

df = pd.read_csv('/Users/aimeeco/peer-review-ML-model/data/processed_data.csv')

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

# use MultiLabelBinarizer to convert labels to multi-hot encoded format
mlb = MultiLabelBinarizer()
encoded_labels = mlb.fit_transform(labels)

assert len(texts) == len(encoded_labels), f"The number of texts ({len(texts)}) and labels ({len(encoded_labels)}) should match"

# tokenize data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)

class PaperDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)  
        return item
    
    def __len__(self):
        return len(self.labels)

dataset = PaperDataset(encodings, encoded_labels)
train_dataset, val_dataset = train_test_split(dataset, test_size=0.2)

# load model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(mlb.classes_))  

# training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()
