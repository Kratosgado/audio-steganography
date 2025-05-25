import torch
import torch.nn as nn

class AudioStegEncoder(nn.Module):
    def __init__(self):
        super(AudioStegEncoder, self).__init__()
        self.conv1 = nn.Conv1d(2, 16, kernel_size=9, padding=4)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(16, 1, kernel_size=9, padding=4)

    def forward(self, audio, message):
        x = torch.cat([audio, message], dim=1)  # [B, 2, T]
        x = self.relu(self.conv1(x))
        stego = self.conv2(x)  # [B, 1, T]
        return stego
