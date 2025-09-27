import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class TextClassifier(nn.Module):
    def __init__(self, base_model, hidden_dim=256, num_classes=2):
        super(TextClassifier, self).__init__()
        self.bert = base_model
        self.fc1 = nn.Linear(768, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state.mean(dim=1)
        x = self.relu(self.fc1(pooled))
        return self.fc2(x)

def train_text_encoder(csv_path="data/dataset.csv", save_path="models/text_encoder.pt"):
    df = pd.read_csv(csv_path)  # dataset: ["text", "label"]
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    labels = LabelEncoder().fit_transform(df["label"])

    X_train, X_test, y_train, y_test = train_test_split(df["text"], labels, test_size=0.2)

    model = TextClassifier(DistilBertModel.from_pretrained("distilbert-base-uncased"))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    model.train()
    for epoch in range(3):
        for text, label in zip(X_train, y_train):
            encodings = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            outputs = model(encodings["input_ids"], encodings["attention_mask"])
            loss = criterion(outputs, torch.tensor([label]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}: Loss={loss.item()}")

    torch.save(model, save_path)
    print(f"✅ Text encoder saved at {save_path}")
