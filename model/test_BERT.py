import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import MultiLabelBinarizer
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

# Check device availability
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# Load your test dataset
test_df = pd.read_csv('/Users/aimeeco/peer-review-ML-model/data/processed_data.csv')

# Ensure 'normalized_sentence' is loaded as a list of strings
test_texts = test_df['normalized_sentence'].tolist()

# Filter out non-string elements
test_texts = [text for text in test_texts if isinstance(text, str)]

# Verify the content of texts to ensure it's a list of strings
print(f"Number of texts: {len(test_texts)}")
print(f"Sample texts: {test_texts[:5]}")

# Filter the original DataFrame to match the filtered texts
test_df_filtered = test_df[test_df['normalized_sentence'].apply(lambda x: isinstance(x, str))].copy()

# Convert labels to lists and ensure they are strings
def convert_labels_to_list(label):
    if isinstance(label, float):
        return []
    label_string = str(label).strip("[]").replace("'", "")
    return [lbl.strip() for lbl in label_string.split(',')]

test_df_filtered.loc[:, 'section_label'] = test_df_filtered['section_label'].apply(convert_labels_to_list)
test_df_filtered.loc[:, 'aspect_label'] = test_df_filtered['aspect_label'].apply(convert_labels_to_list)
test_df_filtered.loc[:, 'purpose_label'] = test_df_filtered['purpose_label'].apply(convert_labels_to_list)
test_df_filtered.loc[:, 'significance_label'] = test_df_filtered['significance_label'].apply(convert_labels_to_list)

# Combine labels into a single list
test_labels = test_df_filtered[['section_label', 'aspect_label', 'purpose_label', 'significance_label']].apply(lambda row: row.iloc[0] + row.iloc[1] + row.iloc[2] + row.iloc[3], axis=1).tolist()

# Use MultiLabelBinarizer to convert labels to multi-hot encoded format
mlb = MultiLabelBinarizer()
encoded_test_labels = mlb.fit_transform(test_labels)

# Ensure that the length of texts and labels match
assert len(test_texts) == len(encoded_test_labels), f"The number of texts ({len(test_texts)}) and labels ({len(encoded_test_labels)}) should match"

# Tokenize the test data
tokenizer = BertTokenizer.from_pretrained('./results')
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

# Create test dataset
test_dataset = TestDataset(test_encodings, encoded_test_labels)

# Load the trained model from the checkpoint
model = BertForSequenceClassification.from_pretrained('./results/checkpoint-10360')
model.to(device)

# Predict
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
        raw_logits.extend(logits.cpu().numpy())  # Collect raw logits
        probs = torch.sigmoid(logits).cpu().numpy()
        predicted_probs.extend(probs)
        true_labels.extend(labels.cpu().numpy())

# Convert predicted_probs to a NumPy array
predicted_probs = np.array(predicted_probs)

# Print raw logits for inspection
print("Sample raw logits:")
print(raw_logits[:5])

# Experiment with different thresholds
threshold = 0.1
predicted_labels = (predicted_probs > threshold).astype(int)

# Convert predicted labels back to original label format
predicted_labels = mlb.inverse_transform(predicted_labels)

# Calculate evaluation metrics
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

precision, recall, f1, _ = precision_recall_fscore_support(encoded_test_labels, (predicted_probs > threshold).astype(int), average='micro')
accuracy = accuracy_score(encoded_test_labels, (predicted_probs > threshold).astype(int))

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")
print(f"Accuracy: {accuracy}")

# Print a few predictions with the adjusted threshold
for i in range(5):
    print(f"Text: {test_texts[i]}")
    print(f"True Labels: {test_labels[i]}")
    print(f"Predicted Labels: {predicted_labels[i]}")
    print()
