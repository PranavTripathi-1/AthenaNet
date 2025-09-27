import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import os

class AudioEncoderModel(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=256):
        super(AudioEncoderModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

def extract_features(file_path):
    waveform, sr = torchaudio.load(file_path)
    mel_spec = torchaudio.transforms.MelSpectrogram()(waveform)
    return mel_spec.mean(dim=-1).squeeze(0)[:128]  # fixed size vector

def train_audio_encoder(data_dir="data/", save_path="models/audio_encoder.pt"):
    model = AudioEncoderModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    files = [f"data/sample_{i}.wav" for i in range(5)]

    model.train()
    for epoch in range(3):
        total_loss = 0
        for f in files:
            features = extract_features(os.path.join(data_dir, f))
            features = features.unsqueeze(0)
            output = model(features)
            loss = criterion(output, features)  # autoencoder-like training
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss={total_loss/len(files)}")

    torch.save(model, save_path)
    print(f"✅ Audio encoder saved at {save_path}")
