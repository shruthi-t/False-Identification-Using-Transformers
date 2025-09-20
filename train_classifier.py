import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments

# ----------------------
# Paths
# ----------------------
DATA_PATH = "data/data.csv"
OUTPUT_DIR = "outputs/classifier"

# ----------------------
# Load dataset
# ----------------------
df = pd.read_csv(DATA_PATH)

# Map string labels to IDs
label2id = {"manipulation": 0, "impersonation": 1, "trust_building": 2, "distraction": 3}
id2label = {v: k for k, v in label2id.items()}
df["label"] = df["label"].map(label2id)

# Train-validation split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"].tolist(), df["label"].tolist(), test_size=0.2, random_state=42
)

# ----------------------
# Dataset class
# ----------------------
class FraudDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encodings = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {key: val.squeeze(0) for key, val in encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

# ----------------------
# Tokenizer and model
# ----------------------
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=4, id2label=id2label, label2id=label2id
)

# ----------------------
# Prepare datasets
# ----------------------
train_dataset = FraudDataset(train_texts, train_labels, tokenizer)
val_dataset = FraudDataset(val_texts, val_labels, tokenizer)

# ----------------------
# Compute metrics
# ----------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, average="weighted"),
        "recall": recall_score(labels, predictions, average="weighted"),
        "f1": f1_score(labels, predictions, average="weighted"),
    }

# ----------------------
# Training setup
# ----------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-5,  # smaller LR for stability
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=8,  # train longer
    weight_decay=0.01,
    logging_dir="./logs",
    load_best_model_at_end=True,   # keep best checkpoint
    metric_for_best_model="f1",    # choose by F1
    greater_is_better=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# ----------------------
# Train and Save
# ----------------------
trainer.train()
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ Model trained and saved at", OUTPUT_DIR)
