import torch
from .fusion_model import FusionClassifier

class Predictor:
    def __init__(self, model_path="models/fusion_classifier.pt"):
        self.model = FusionClassifier()
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()

    def predict(self, text_features, audio_features):
        with torch.no_grad():
            probs = self.model(text_features, audio_features)
        confidence, pred_class = torch.max(probs, dim=1)
        risk_level = "High Risk" if pred_class.item() == 1 else "Low Risk"
        return risk_level, confidence.item()
