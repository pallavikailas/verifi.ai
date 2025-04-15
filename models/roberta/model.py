import torch
import torch.nn as nn
from transformers import RobertaModel

class RoBERTaClassifier(nn.Module):
    def __init__(self):
        super(RoBERTaClassifier, self).__init__()
        # Load the pre-trained RoBERTa model
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.classifier = nn.Sequential(
            nn.Linear(self.roberta.config.hidden_size, 256),  # The output dimension of RoBERTa is 768, use it accordingly
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)  # Binary classification, so output dimension is 2
        )

    def forward(self, input_ids, attention_mask):
        # Pass through RoBERTa model
        output = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        # Use the pooler_output from RoBERTa's final layer
        return self.classifier(output.pooler_output)
