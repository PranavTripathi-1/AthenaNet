import torch
import os
import torchaudio
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from src.infrastructure.audio_encoder import AudioEncoder

def evaluate_audio_encoder(model_path="models/audio_encoder.pt", csv_path="data/dataset.csv"):
    df = pd.read_csv(csv_path)  # ["audio_path", "label"]
    labels = LabelEncoder().fit_transform(df["label"])

    X_train, X_test, y_train, y_test = train_test_split(df["audio_path"], labels, test_size=0.2)

    model = AudioEncoder(model_path)
    y_pred = []

    with torch.no_grad():
        for audio_file in X_test:
            feat = model.encode(audio_file)
            pred = torch.argmax(feat).item()  # Example heuristic
            y_pred.append(pred)

    print("📊 Audio Encoder Evaluation")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    evaluate_audio_encoder()
