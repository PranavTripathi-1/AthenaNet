# src/infrastructure/audio_encoder.py
import torch
import torch.nn as nn
import torchaudio

class AudioEncoder(nn.Module):
    def __init__(self, model_path=None, embedding_dim=128):
        super(AudioEncoder, self).__init__()
        # Simple CNN for audio embeddings
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(64, embedding_dim)
        
        if model_path is not None:
            self.load_state_dict(torch.load(model_path))
        self.eval()
    
    def forward(self, waveform):
        """
        waveform: torch.Tensor of shape [batch_size, 1, time]
        """
        x = self.cnn(waveform)
        x = x.squeeze(-1)  # shape: [batch_size, 64]
        embeddings = self.fc(x)
        return embeddings

    def encode(self, audio_path):
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = waveform.mean(dim=0, keepdim=True)  # convert to mono
        waveform = waveform.unsqueeze(0)  # batch dimension
        with torch.no_grad():
            embedding = self.forward(waveform)
        return embedding
