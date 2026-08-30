import streamlit as st
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import pdfplumber

# Model Class - same as train.py
class BertClassifier(nn.Module):
    def __init__(self, num_classes):
        super(BertClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        self.drop = nn.Dropout(p=0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.drop(pooled_output)
        return self.fc(output)

# Load model
@st.cache_resource
def load_model():
    df = pd.read_csv('resume 1.csv')
    categories = df['Category'].unique()
    label_to_id = {label: i for i, label in enumerate(categories)}
    
    model = BertClassifier(num_classes=len(categories))
    model.load_state_dict(torch.load('resume_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model, label_to_id

model, label_to_id = load_model()
id_to_label = {v: k for k, v in label_to_id.items()}
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Streamlit UI
st.title("📄 Resume Category Predictor")
st.write("Upload a Resume PDF and I will predict the Job Category")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    with pdfplumber.open(uploaded_file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    
    st.text_area("Extracted Text", text[:1000], height=200)
    
    if st.button("Predict Category"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(inputs['input_ids'], inputs['attention_mask'])
            _, predicted = torch.max(outputs, 1)
            prediction = id_to_label[predicted.item()]
        
        st.success(f"### Predicted Category: {prediction}")