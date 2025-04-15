import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer_name='bert-base-uncased', max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        encoding = self.tokenizer(text, padding='max_length', truncation=True,
                                  max_length=self.max_len, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': label
        }

def load_data(fake_path, true_path, test_size=0.2):
    fake = pd.read_csv(fake_path)
    true = pd.read_csv(true_path)
    fake['label'] = 0
    true['label'] = 1
    df = pd.concat([fake, true]).sample(frac=1).reset_index(drop=True)
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test
