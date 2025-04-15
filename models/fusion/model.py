import torch
import torch.nn as nn

class FusionModel(nn.Module):
    def __init__(self, text_model, metadata_dim=3):
        super(FusionModel, self).__init__()
        # Text model, e.g., RoBERTa or BERT
        self.text_model = text_model
        # Fully connected layers after combining text output and metadata
        self.fc = nn.Sequential(
            nn.Linear(2 + metadata_dim, 128),  # 2 is for the text output size (for binary classification), metadata_dim is the additional features
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)  # Binary classification (fake vs real news)
        )

    def forward(self, input_ids, attention_mask, metadata):
        # Extract text output using the provided text model (e.g., RoBERTa)
        text_output = self.text_model(input_ids, attention_mask)
        # Concatenate the text output with the metadata
        out = torch.cat((text_output, metadata), dim=1)
        # Pass through the fully connected layers
        return self.fc(out)
