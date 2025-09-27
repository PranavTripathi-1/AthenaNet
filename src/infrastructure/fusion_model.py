import torch
import torch.nn as nn

class FusionClassifier(nn.Module):
    def __init__(self, input_dim_text=768, input_dim_audio=256, hidden_dim=512, num_classes=2):
        super(FusionClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim_text + input_dim_audio, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, text_features, audio_features):
        combined = torch.cat((text_features, audio_features), dim=1)
        x = self.relu(self.fc1(combined))
        x = self.fc2(x)
        return self.softmax(x)
