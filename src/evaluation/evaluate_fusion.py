import torch
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from src.infrastructure.text_encoder import TextEncoder
from src.infrastructure.audio_encoder import AudioEncoder
from src.infrastructure.fusion_model import FusionClassifier

def evaluate_fusion(model_path="models/fusion_classifier.pt", csv_path="data/dataset.csv"):
    df = pd.read_csv(csv_path)  # ["text", "audio_path", "label"]
    labels = LabelEncoder().fit_transform(df["label"])

    X_train, X_test, y_train, y_test = train_test_split(df, labels, test_size=0.2)

    text_encoder = TextEncoder()
    audio_encoder = AudioEncoder()

    model = FusionClassifier()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    y_pred = []
    with torch.no_grad():
        for _, row in X_test.iterrows():
            text_feat = text_encoder.encode(row["text"])
            audio_feat = audio_encoder.encode(row["audio_path"])
            outputs = model(text_feat, audio_feat)
            pred = outputs.argmax(dim=1).item()
            y_pred.append(pred)

    print("📊 Fusion Model Evaluation")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    evaluate_fusion()
