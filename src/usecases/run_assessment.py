# src/usecases/run_assessment.py
from src.infrastructure.predictor import Predictor

def run_assessment(text_input, audio_path):
    """
    Run a single assessment with text and audio input.
    Returns the risk score (0-1)
    """
    predictor = Predictor()
    risk_score = predictor.predict(text_input, audio_path)
    return risk_score

# Example standalone usage
if __name__ == "__main__":
    sample_text = "I have been feeling anxious and sad lately."
    sample_audio = "data/sample.wav"
    score = run_assessment(sample_text, sample_audio)
    print(f"Predicted Risk Score: {score}")
