import torch.nn as nn
from transformers import GPT2Model

class GPT2Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.gpt2 = GPT2Model.from_pretrained("gpt2")
        self.fc = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Linear(256, 2))

    def forward(self, input_ids, attention_mask):
        out = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        return self.fc(out.last_hidden_state[:, -1, :])
