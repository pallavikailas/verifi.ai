from transformers import Trainer, TrainingArguments
from datasets import Dataset

def train_model(model, train_texts, train_labels):
    dataset = Dataset.from_dict({'text': train_texts, 'label': train_labels})
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def tokenize(batch):
        return tokenizer(batch['text'], padding='max_length', truncation=True)

    dataset = dataset.map(tokenize, batched=True)
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        per_device_train_batch_size=8,
        num_train_epochs=3,
        logging_dir="./logs",
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
    trainer.train()
