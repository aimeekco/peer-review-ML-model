import pandas as pd
import transformers
from sklearn.preprocessing import MultiLabelBinarizer
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

# check device availability
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

test_df = pd.read_csv('/Users/aimeeco/peer-review-ML-model/data/processed_data.csv')

test_texts = test_df['normalized_sentence'].tolist()

test_texts = [text for text in test_texts if isinstance(text, str)]
print(f"Number of texts: {len(test_texts)}")
print(f"Sample texts: {test_texts[:5]}")

test_df_filtered = test_df[test_df['normalized_sentence'].apply(lambda x: isinstance(x, str))].copy()

def convert_labels_to_list(label):
    if isinstance(label, float):
        return []
    label_string = str(label).strip("[]").replace("'", "")
    return [lbl.strip() for lbl in label_string.split(',')]

test_df_filtered.loc[:, 'section_label'] = test_df_filtered['section_label'].apply(convert_labels_to_list)
test_df_filtered.loc[:, 'aspect_label'] = test_df_filtered['aspect_label'].apply(convert_labels_to_list)
test_df_filtered.loc[:, 'purpose_label'] = test_df_filtered['purpose_label'].apply(convert_labels_to_list)
test_df_filtered.loc[:, 'significance_label'] = test_df_filtered['significance_label'].apply(convert_labels_to_list)

test_labels = test_df_filtered[['section_label', 'aspect_label', 'purpose_label', 'significance_label']].apply(lambda row: row.iloc[0] + row.iloc[1] + row.iloc[2] + row.iloc[3], axis=1).tolist()

mlb = MultiLabelBinarizer()
encoded_test_labels = mlb.fit_transform(test_labels)

assert len(test_texts) == len(encoded_test_labels), f"The number of texts ({len(test_texts)}) and labels ({len(encoded_test_labels)}) should match"

tokenizer = transformers.LlamaTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B')
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512)

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]).to(device) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float).to(device)
        return item
    
    def __len__(self):
        return len(self.labels)

test_dataset = TestDataset(test_encodings, encoded_test_labels)

model = transformers.LlamaForSequenceClassification.from_pretrained('meta-llama/Meta-Llama-3-8B', num_labels=len(mlb.classes_), torch_dtype=torch.bfloat16, device_map="auto")
model.to(device)

model.eval()
predicted_probs = []
true_labels = []
raw_logits = []

with torch.no_grad():
    for batch in DataLoader(test_dataset, batch_size=8):
        inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
        labels = batch['labels'].to(device)
        outputs = model(**inputs)
        logits = outputs.logits
        raw_logits.extend(logits.cpu().numpy()) 
        probs = torch.sigmoid(logits).cpu().numpy()
        predicted_probs.extend(probs)
        true_labels.extend(labels.cpu().numpy())

predicted_probs = np.array(predicted_probs)

print("Sample raw logits:")
print(raw_logits[:5])

threshold = 0.1
predicted_labels = (predicted_probs > threshold).astype(int)

predicted_labels = mlb.inverse_transform(predicted_labels)

from sklearn.metrics import precision_recall_fscore_support, accuracy_score

precision, recall, f1, _ = precision_recall_fscore_support(encoded_test_labels, (predicted_probs > threshold).astype(int), average='micro')
accuracy = accuracy_score(encoded_test_labels, (predicted_probs > threshold).astype(int))

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")
print(f"Accuracy: {accuracy}")

for i in range(5):
    print(f"Text: {test_texts[i]}")
    print(f"True Labels: {test_labels[i]}")
    print(f"Predicted Labels: {predicted_labels[i]}")
    print()
