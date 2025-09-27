from dataclasses import dataclass

@dataclass
class TextInput:
    content: str

@dataclass
class AudioInput:
    file_path: str

@dataclass
class PredictionResult:
    risk_level: str
    confidence: float
    explanation: str
