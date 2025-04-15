import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
from torch.optim import AdamW
from tqdm import tqdm
from models.roberta.model import RoBERTaClassifier
from dataloader.news_loader import load_data, NewsDataset
from torch.utils.data import DataLoader

def train_roberta_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}.")

    X_train, _, y_train, _ = load_data("data/raw/Fake.csv", "data/raw/True.csv")
    
    train_dataset = NewsDataset(X_train, y_train, "roberta-base")
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    model = RoBERTaClassifier().to(device)
    
    optimizer = AdamW(model.parameters(), lr=2e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
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
    print("Model saved to models/roberta/roberta_model.pt")
