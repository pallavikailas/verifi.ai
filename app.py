import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch

st.title("📰 Fake News Detector")
text_input = st.text_area("Enter the news content")

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

if st.button("Detect"):
    tokens = tokenizer(text_input, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**tokens).logits
    prediction = torch.argmax(logits).item()
    st.success("✅ Real News" if prediction == 0 else "🚨 Fake News")
