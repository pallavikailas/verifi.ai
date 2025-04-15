import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
from torch.optim import AdamW
from tqdm import tqdm
from models.roberta.model import RoBERTaClassifier  # or another text model (e.g., BERT)
from models.fusion.model import FusionModel
from dataloader.news_loader import load_data, NewsDataset
from torch.utils.data import DataLoader
import os

def train_fusion_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the dataset, which should also return metadata (e.g., 3 features)
    X_train, _, y_train, metadata_train = load_dataset("data/raw/fake.zip", "data/raw/true.zip", return_metadata=True)
    
    # Assuming your dataset is ready to return the metadata along with the text
    train_dataset = NewsDataset(X_train, y_train, metadata_train, tokenizer_name="roberta-base")
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # Load the base RoBERTa model
    text_model = RoBERTaClassifier().to(device)

    # Create the Fusion model using the text model and metadata
    fusion_model = FusionModel(text_model).to(device)
    optimizer = AdamW(fusion_model.parameters(), lr=2e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    fusion_model.train()
    for epoch in range(3):
        total_loss = 0
        for batch in tqdm(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            metadata = batch['metadata'].to(device)  # Metadata features

            optimizer.zero_grad()
            outputs = fusion_model(input_ids, attention_mask, metadata)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} loss: {total_loss:.4f}")

    # Save the trained fusion model
    os.makedirs("models/fusion", exist_ok=True)
    torch.save(fusion_model.state_dict(), "models/fusion/fusion_model.pt")

