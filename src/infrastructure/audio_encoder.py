import torch
import torchaudio

class AudioEncoder:
    def __init__(self, model_path="models/audio_encoder.pt"):
        self.model = torch.load(model_path, map_location="cpu")
        self.model.eval()

    def encode(self, file_path: str):
        waveform, sr = torchaudio.load(file_path)
        with torch.no_grad():
            embedding = self.model(waveform)
        return embedding
