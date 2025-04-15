import streamlit as st
import torch
from transformers import AutoTokenizer
from models.bert.model import BERTClassifier

@st.cache_resource
def load_model():
    model = BERTClassifier()
    model.load_state_dict(torch.load("models/bert/bert_model.pt"))
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    return model, tokenizer

model, tokenizer = load_model()

st.title("🧠 Verifi.ai Pro")
news = st.text_area("Enter a news headline or story:")
if st.button("Predict"):
    tokens = tokenizer(news, return_tensors='pt', truncation=True, padding='max_length', max_length=256)
    with torch.no_grad():
        output = model(tokens['input_ids'], tokens['attention_mask'])
        pred = torch.argmax(output, dim=1).item()
        st.success("✅ REAL News" if pred == 1 else "🚨 FAKE News")
