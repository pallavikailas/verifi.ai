import torch
from transformers import AutoTokenizer
from models.bert.model import BERTClassifier

# Load model and tokenizer once
model = BERTClassifier()
model.load_state_dict(torch.load("models/bert/bert_model.pt", map_location=torch.device('cpu')))
model.eval()
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def predict_news(text):
    encoded = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=256)
    with torch.no_grad():
        output = model(encoded['input_ids'], encoded['attention_mask'])
        prediction = torch.argmax(output, dim=1).item()
    return "REAL" if prediction == 1 else "FAKE"
