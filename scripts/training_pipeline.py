import os
import sys

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torch.optim import AdamW
from sklearn.metrics import accuracy_score
from tqdm import tqdm


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.bert.model import BERTClassifier
from models.gpt2.model import GPT2Classifier
from models.roberta.model import RoBERTaClassifier
from dataloader.news_loader import NewsDataset, load_data

# Paths
FAKE_PATH = "data/raw/fake.csv"
TRUE_PATH = "data/raw/true.csv"

# Common
BATCH_SIZE = 8
EPOCHS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
X_train, X_test, y_train, y_test = load_dataset(FAKE_PATH, TRUE_PATH)

# Tokenizer map for each model
tokenizer_map = {
    "bert": "bert-base-uncased",
    "gpt2": "gpt2",
    "roberta": "roberta-base"
}

def train_and_save(model_name, ModelClass, model_path):
    print(f"\n🔧 Training {model_name.upper()}...")
    tokenizer_name = tokenizer_map[model_name]
    train_dataset = FakeNewsDataset(X_train, y_train, tokenizer_name)
    test_dataset = FakeNewsDataset(X_test, y_test, tokenizer_name)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = ModelClass().to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"{model_name.upper()} Epoch {epoch+1}"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"{model_name.upper()} Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")

    # Evaluation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"{model_name.upper()} Test Accuracy: {acc:.4f}")

    # Save
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"{model_name.upper()} model saved to {model_path}")

# Train all models
train_and_save("bert", BERTClassifier, "models/bert/bert_model.pt")
train_and_save("gpt2", GPT2Classifier, "models/gpt2/gpt2_model.pt")
train_and_save("roberta", RoBERTaClassifier, "models/roberta/roberta_model.pt")
