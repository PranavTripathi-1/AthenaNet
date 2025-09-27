import torch
from src.infrastructure.text_encoder import TextEncoder
from src.infrastructure.audio_encoder import AudioEncoder
from src.infrastructure.predictor import Predictor
from src.infrastructure.llm_integration import LLMExplainer
from src.domain.entities import TextInput, AudioInput, PredictionResult

def run_full_assessment(user_text: str, audio_file):
    text_encoder = TextEncoder()
    audio_encoder = AudioEncoder()
    predictor = Predictor()
    explainer = LLMExplainer()

    text_features = text_encoder.encode(user_text) if user_text else torch.zeros((1,768))
    audio_features = audio_encoder.encode("data/sample.wav") if audio_file else torch.zeros((1,256))

    risk_level, confidence = predictor.predict(text_features, audio_features)
    explanation = explainer.explain_prediction(user_text, risk_level)

    return PredictionResult(risk_level, confidence, explanation).__dict__
