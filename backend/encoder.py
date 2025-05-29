import torch
import torch.nn as nn

class AudioStegEncoder(nn.Module):
    # def __init__(self):
    #     super(ImprovedAudioStegEncoder, self).__init__()
    #     # More sophisticated architecture
    #     self.conv1 = nn.Conv1d(2, 32, kernel_size=15, padding=7)
    #     self.conv2 = nn.Conv1d(32, 64, kernel_size=15, padding=7)
    #     self.conv3 = nn.Conv1d(64, 32, kernel_size=15, padding=7)
    #     self.conv4 = nn.Conv1d(32, 1, kernel_size=15, padding=7)
    #     self.relu = nn.ReLU()
    #     self.dropout = nn.Dropout(0.1)
        
    # def forward(self, audio, message):
    #     # Ensure both inputs have the same length
    #     min_len = min(audio.size(-1), message.size(-1))
    #     audio = audio[:, :, :min_len]
    #     message = message[:, :, :min_len]
        
    #     x = torch.cat([audio, message], dim=1)  # [B, 2, T]
        
    #     x = self.relu(self.conv1(x))
    #     x = self.dropout(x)
    #     x = self.relu(self.conv2(x))
    #     x = self.dropout(x)
    #     x = self.relu(self.conv3(x))
    #     stego = self.conv4(x)  # [B, 1, T]
        
        # Add residual connection with original audio
        # return audio + 0.01 * stego  # Small perturbation

    def __init__(self):
        super(AudioStegEncoder, self).__init__()
        self.conv1 = nn.Conv1d(2, 16, kernel_size=9, padding=4)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(16, 1, kernel_size=9, padding=4)

    def forward(self, audio, message):
        # Ensure both inputs have the same length
        if message.dim() == 2:
            message = message.unsqueeze(1)  # Add channel dimension if missing
            
        # Make sure audio has the right shape [B, C, T]
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
            
        # Adjust message length to match audio
        if message.size(-1) != audio.size(-1):
            if message.size(-1) < audio.size(-1):
                # Repeat message to match audio length
                repeat_factor = audio.size(-1) // message.size(-1) + 1
                message = message.repeat(1, 1, repeat_factor)[:, :, :audio.size(-1)]
            else:
                # Truncate message to match audio length
                message = message[:, :, :audio.size(-1)]
        
        x = torch.cat([audio, message], dim=1)  # [B, 2, T]
        x = self.relu(self.conv1(x))
        stego_delta = self.conv2(x)  # [B, 1, T]
        
        # Add residual connection with original audio (small perturbation)
        return audio + 0.01 * stego_delta
