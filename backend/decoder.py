import torch
import torch.nn as nn

class AudioStegDecoder(nn.Module):
    def __init__(self):
        super(AudioStegDecoder, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=9, padding=4)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(16, 1, kernel_size=9, padding=4)
        self.sigmoid = nn.Sigmoid()

    def forward(self, stego_audio):
        x = self.relu(self.conv1(stego_audio))
        x = self.sigmoid(self.conv2(x))
        return x
