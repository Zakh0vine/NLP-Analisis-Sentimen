import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import CrossEntropyLoss

# Cek CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Load Dataset
file_path = "LOKASI DATASETNYA, Contoh : C:/Documents/Folder1/namadataset.csv"
data = pd.read_csv(file_path)

# 2. Pra-pemrosesan Data
data = data[['text', 'sentiment']].dropna()
# Map sentimen menjadi label numerik
data['sentiment'] = data['sentiment'].map({'positive': 2, 'negative': 0, 'neutral': 1})  # Positive=2, Neutral=1, Negative=0

# Split data
train_texts, val_texts, train_labels, val_labels = train_test_split(
    data['text'], data['sentiment'], test_size=0.2, random_state=42
)

# 3. Tokenisasi
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class SentimentDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        self.encodings = tokenizer(list(texts), truncation=True, padding=True, max_length=128)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.encodings['input_ids'][idx]),
            'attention_mask': torch.tensor(self.encodings['attention_mask'][idx]),
            'labels': torch.tensor(self.labels.iloc[idx])
        }

train_dataset = SentimentDataset(train_texts, train_labels)
val_dataset = SentimentDataset(val_texts, val_labels)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# 4. Model BERT
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
model = model.to(device)

# 5. Optimizer dan Loss Function
optimizer = AdamW(model.parameters(), lr=5e-5)
loss_fct = CrossEntropyLoss()  # CrossEntropyLoss untuk multi-kelas

# 6. Training Loop
epochs = 3
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device).long()  # Mengubah tipe ke Long tensor
        attention_mask = batch['attention_mask'].to(device).float()  # Attention mask ke float
        labels = batch['labels'].to(device).long()  # Labels ke Long tensor

        # Proses forward
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # Menghitung loss
        loss = loss_fct(logits, labels)
        total_loss += loss.item()
        # Backward pass and optimizer step
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")

# 7. Evaluasi Model
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Print classification report
report = classification_report(all_labels, all_preds, target_names=["Negative", "Neutral", "Positive"], output_dict=True)
print(classification_report(all_labels, all_preds, target_names=["Negative", "Neutral", "Positive"]))

# 8. Menampilkan dan Menyimpan Grafik Precision, Recall, F1-Score
precision = [report[class_name]['precision'] for class_name in ["Negative", "Neutral", "Positive"]]
recall = [report[class_name]['recall'] for class_name in ["Negative", "Neutral", "Positive"]]
f1_score = [report[class_name]['f1-score'] for class_name in ["Negative", "Neutral", "Positive"]]

labels = ["Negative", "Neutral", "Positive"]
x = np.arange(len(labels))

fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.2
ax.bar(x - bar_width, precision, bar_width, label="Precision")
ax.bar(x, recall, bar_width, label="Recall")
ax.bar(x + bar_width, f1_score, bar_width, label="F1-Score")

ax.set_xlabel("Classes")
ax.set_ylabel("Scores")
ax.set_title("Precision, Recall, and F1-Score per Class")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Simpan bar chart sebagai file gambar
plt.savefig("precision_recall_f1.png")
plt.show()

# Menyimpan model dan tokenizer
model_save_path = "Z:/Documents/RCNN-Model/Results"
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

print(f"Model dan tokenizer disimpan di {model_save_path}")
