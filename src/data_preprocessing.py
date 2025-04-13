import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

def load_data():
    df = pd.read_csv("data/fake.csv") 
    df = df[['title', 'text', 'label']].dropna()
    df['label'] = df['label'].map({'REAL': 0, 'FAKE': 1})
    return train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

def tokenize_data(texts, tokenizer_name='bert-base-uncased', max_length=512):
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    return tokenizer(list(texts), padding=True, truncation=True, return_tensors='pt', max_length=max_length)
