from transformers import BertForSequenceClassification

def build_model():
    return BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
