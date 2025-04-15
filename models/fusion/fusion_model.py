import torch.nn as nn

class FusionModel(nn.Module):
    def __init__(self, text_model, metadata_dim=3):
        super(FusionModel, self).__init__()
        self.text_model = text_model
        self.fc = nn.Sequential(
            nn.Linear(2 + metadata_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )

    def forward(self, input_ids, attention_mask, metadata):
        text_output = self.text_model(input_ids, attention_mask)
        out = torch.cat((text_output, metadata), dim=1)
        return self.fc(out)
