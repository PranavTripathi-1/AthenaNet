# src/infrastructure/fusion_model.py
import torch
import torch.nn as nn

class FusionClassifier(nn.Module):
    def __init__(self, text_embedding_dim=768, audio_embedding_dim=128, hidden_dim=256, model_path=None):
        super(FusionClassifier, self).__init__()
        # Fusion MLP
        self.fc = nn.Sequential(
            nn.Linear(text_embedding_dim + audio_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        if model_path is not None:
            self.load_state_dict(torch.load(model_path))
        self.eval()

    def forward(self, text_emb, audio_emb):
        """
        text_emb: torch.Tensor [batch_size, text_embedding_dim]
        audio_emb: torch.Tensor [batch_size, audio_embedding_dim]
        """
        fused = torch.cat([text_emb, audio_emb], dim=1)
        score = self.fc(fused)
        return score

    def predict(self, text_emb, audio_emb):
        with torch.no_grad():
            return self.forward(text_emb, audio_emb).item()
