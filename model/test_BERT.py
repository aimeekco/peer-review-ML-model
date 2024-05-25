import pandas as pd
from transformers import BertTokenizer, Trainer, BertForSequenceClassification
from sklearn.preprocessing import MultiLabelBinarizer
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# Load your test dataset
test_df = pd.read_csv('/Users/aimeeco/peer-review-ML-model/data/processed_data.csv')

# Ensure 'normalized_sentence' is loaded as a list of strings
test_texts = test_df['normalized_sentence'].tolist()

# Filter out non-string elements
test_texts = [text for text in test_texts if isinstance(text, str)]

# Verify the content of texts to ensure it's a list of strings
print(f"Number of texts: {len(test_texts)}")
print(f"Sample texts: {test_texts[10:15]}")

# Filter the original DataFrame to match the filtered texts
test_df_filtered = test_df[test_df['normalized_sentence'].apply(lambda x: isinstance(x, str))]

# Convert labels to lists and ensure they are strings
def convert_labels_to_list(label):
    if isinstance(label, float):
        return []
    label_string = str(label).strip("[]").replace("'", "")
    return [lbl.strip() for lbl in label_string.split(',')]

test_df_filtered['section_label'] = test_df_filtered['section_label'].apply(convert_labels_to_list)
test_df_filtered['aspect_label'] = test_df_filtered['aspect_label'].apply(convert_labels_to_list)
test_df_filtered['purpose_label'] = test_df_filtered['purpose_label'].apply(convert_labels_to_list)
test_df_filtered['significance_label'] = test_df_filtered['significance_label'].apply(convert_labels_to_list)

# Combine labels into a single list
test_labels = test_df_filtered[['section_label', 'aspect_label', 'purpose_label', 'significance_label']].apply(lambda row: row.iloc[0] + row.iloc[1] + row.iloc[2] + row.iloc[3], axis=1).tolist()

# Use MultiLabelBinarizer to convert labels to multi-hot encoded format
mlb = MultiLabelBinarizer()
encoded_test_labels = mlb.fit_transform(test_labels)

# Ensure that the length of texts and labels match
assert len(test_texts) == len(encoded_test_labels), f"The number of texts ({len(test_texts)}) and labels ({len(encoded_test_labels)}) should match"

# Tokenize the test data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512)

class TestDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)  # Ensure labels are of float type for multi-label classification
        return item
    
    def __len__(self):
        return len(self.labels)

# Create test dataset
test_dataset = TestDataset(test_encodings, encoded_test_labels)

# Create a DataLoader for the test dataset
test_loader = DataLoader(test_dataset, batch_size=8)

# Load the trained model
model = BertForSequenceClassification.from_pretrained('./results/checkpoint-6000')

# Initialize Trainer
trainer = Trainer(
    model=model
)

predictions = trainer.predict(test_dataset=test_dataset)

predicted_probs = predictions.predictions

# for i in range(5):
#     print(f"Text: {test_texts[i]}")
#     print(f"True Labels: {test_labels[i]}")
#     print(f"Prediction probabilities: {predicted_probs[i]}")
#     print()
    
    
threshold = 0.2
predicted_labels = (predicted_probs > threshold).astype(int)

predicted_labels = mlb.inverse_transform(predicted_labels)

true_labels = mlb.inverse_transform(encoded_test_labels)

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

