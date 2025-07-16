import torch
import torch.nn as nn

class AudioStegDecoder(nn.Module):
    def __init__(self):
        super(AudioStegDecoder, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=15, padding=7)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=15, padding=7)
        self.conv3 = nn.Conv1d(64, 32, kernel_size=15, padding=7)
        self.conv4 = nn.Conv1d(32, 1, kernel_size=15, padding=7)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.1)

    def forward(self, stego_audio):
        x = self.relu(self.conv1(stego_audio))
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.relu(self.conv3(x))
        x = self.sigmoid(self.conv4(x))
        return x

    # def __init__(self):
    #     super(AudioStegDecoder, self).__init__()
    #     self.conv1 = nn.Conv1d(1, 16, kernel_size=9, padding=4)
    #     self.relu = nn.ReLU()
    #     self.conv2 = nn.Conv1d(16, 1, kernel_size=9, padding=4)
    #     self.sigmoid = nn.Sigmoid()

    # def forward(self, stego_audio):
    #     x = self.relu(self.conv1(stego_audio))
    #     x = self.sigmoid(self.conv2(x))
    #     return x
