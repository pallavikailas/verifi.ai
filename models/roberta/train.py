import torch
from transformers import AdamW
from tqdm import tqdm
from models.roberta.model import RoBERTaClassifier
from scripts.data_loader import load_dataset, FakeNewsDataset
from torch.utils.data import DataLoader
import os

def train_roberta_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train, _, y_train, _ = load_dataset("data/raw/fake.zip", "data/raw/true.zip")
    train_dataset = FakeNewsDataset(X_train, y_train, "roberta-base")
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    model = RoBERTaClassifier().to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(3):
        total_loss = 0
        for batch in tqdm(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} loss: {total_loss:.4f}")

    os.makedirs("models/roberta", exist_ok=True)
    torch.save(model.state_dict(), "models/roberta/roberta_model.pt")
