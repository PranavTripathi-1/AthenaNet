import torch
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from transformers import DistilBertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def evaluate_text_encoder(model_path="models/text_encoder.pt", csv_path="data/dataset.csv"):
    df = pd.read_csv(csv_path)
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    labels = LabelEncoder().fit_transform(df["label"])

    X_train, X_test, y_train, y_test = train_test_split(df["text"], labels, test_size=0.2)

    model = torch.load(model_path)
    model.eval()

    y_pred = []
    with torch.no_grad():
        for text in X_test:
            encodings = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            outputs = model(encodings["input_ids"], encodings["attention_mask"])
            pred = outputs.argmax(dim=1).item()
            y_pred.append(pred)

    print("📊 Text Encoder Evaluation")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    evaluate_text_encoder()
