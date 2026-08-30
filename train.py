import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# 1. DATA LOAD
df = pd.read_csv('resume 1.csv')
print("Columns found:", df.columns.tolist())

# Sirf zaroori 2 column lo
df = df[['Category', 'Resume_str']].dropna()
df.columns = ['Category', 'Resume'] # naam easy kar diye

# 2. LABEL ENCODE
le = LabelEncoder()
df['label'] = le.fit_transform(df['Category'])
num_labels = len(le.classes_)
print("Categories:", le.classes_)
print("Total Samples:", len(df))

# 3. TRAIN TEST SPLIT - FIXED
train_text, val_text, train_label, val_label = train_test_split(
    df['Resume'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42)

print("Train Samples:", len(train_text))
print("Val Samples:", len(val_text))

# 4. DATASET CLASS
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
MAX_LEN = 256

class ResumeDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    
    def __len__(self): 
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = tokenizer.encode_plus(
            self.texts[idx], add_special_tokens=True, max_length=MAX_LEN,
            padding='max_length', truncation=True, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

train_dataset = ResumeDataset(train_text, train_label)
val_dataset = ResumeDataset(val_text, val_label)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# 5. MODEL
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
model = model.to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)

# 6. TRAINING LOOP
EPOCHS = 3
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")

# 7. SAVE MODEL + LABEL ENCODER
torch.save({'model_state': model.state_dict(), 'classes': le.classes_}, 'resume_model.pth')
print("✅ Model saved as resume_model.pth")