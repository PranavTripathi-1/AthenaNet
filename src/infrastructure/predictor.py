# src/infrastructure/predictor.py
from src.infrastructure.text_encoder import TextEncoder
from src.infrastructure.audio_encoder import AudioEncoder
from src.infrastructure.fusion_model import FusionClassifier
import torch

class Predictor:
    def __init__(self, 
                 text_model_path="models/text_encoder.pt",
                 audio_model_path="models/audio_encoder.pt",
                 fusion_model_path="models/fusion_classifier.pt"):
        
        self.text_encoder = TextEncoder(model_path=text_model_path)
        self.audio_encoder = AudioEncoder(model_path=audio_model_path)
        self.fusion_model = FusionClassifier(model_path=fusion_model_path)

    def predict(self, text_input, audio_path):
        """
        text_input: str
        audio_path: path to audio file
        returns: risk_score float
        """
        text_emb = self.text_encoder.encode(text_input)  # [1, 768]
        audio_emb = self.audio_encoder.encode(audio_path)  # [1, 128]
        risk_score = self.fusion_model.predict(text_emb, audio_emb)
        return risk_score
