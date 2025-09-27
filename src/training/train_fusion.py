import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from src.infrastructure.fusion_model import FusionClassifier
from src.infrastructure.text_encoder import TextEncoder
from src.infrastructure.audio_encoder import AudioEncoder

def train_fusion(csv_path="data/dataset.csv", save_path="models/fusion_classifier.pt"):
    df = pd.read_csv(csv_path)  # dataset: ["text", "audio_path", "label"]
    labels = LabelEncoder().fit_transform(df["label"])

    X_train, X_test, y_train, y_test = train_test_split(df, labels, test_size=0.2)

    text_encoder = TextEncoder()
    audio_encoder = AudioEncoder()

    model = FusionClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    model.train()
    for epoch in range(3):
        total_loss = 0
        for _, row in X_train.iterrows():
            text_feat = text_encoder.encode(row["text"])
            audio_feat = audio_encoder.encode(row["audio_path"])
            output = model(text_feat, audio_feat)
            loss = criterion(output, torch.tensor([row["label"]]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss={total_loss/len(X_train)}")

    torch.save(model.state_dict(), save_path)
    print(f"✅ Fusion classifier saved at {save_path}")
