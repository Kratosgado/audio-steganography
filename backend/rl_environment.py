import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import gym
from gym import spaces
from encoder import AudioStegEncoder
from decoder import AudioStegDecoder
from utils import text_to_bits, bits_to_tensor, load_audio, save_audio

class AudioStegEnvironment(gym.Env):
    """Custom Environment for audio steganography using reinforcement learning"""
    
    def __init__(self, audio_path=None, message=None, max_steps=100):
        super(AudioStegEnvironment, self).__init__()
        
        # Initialize encoder and decoder models
        self.encoder = AudioStegEncoder()
        self.decoder = AudioStegDecoder()
        
        # Define action and observation spaces
        # Actions: 
        # 1. Encoding strength (0.001 to 0.1)
        # 2. Frequency band selection (low, mid, high)
        # 3. Position selection (0 to 1, representing percentage through audio)
        self.action_space = spaces.Box(
            low=np.array([0.001, 0, 0]), 
            high=np.array([0.1, 2, 1]), 
            dtype=np.float32
        )
        
        # Observation space: audio features + message features
        self.observation_space = spaces.Dict({
            'audio_features': spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32),
            'message_length': spaces.Discrete(1000),
            'previous_snr': spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
            'previous_bit_accuracy': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        })
        
        # Load default audio and message if provided
        self.default_audio_path = audio_path
        self.default_message = message
        self.max_steps = max_steps
        self.current_step = 0
        
        # Initialize state
        self.reset()
    
    def extract_audio_features(self, audio):
        """Extract relevant features from audio for the observation space"""
        # Convert to mono if stereo
        if audio.dim() > 1 and audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        
        # Extract time domain features
        rms = torch.sqrt(torch.mean(audio**2))
        peak = torch.max(torch.abs(audio))
        
        # Extract frequency domain features using STFT
        n_fft = 1024
        hop_length = 512
        spec = T.Spectrogram(n_fft=n_fft, hop_length=hop_length)(audio)
        
        # Calculate spectral centroid
        freqs = torch.linspace(0, 1, n_fft // 2 + 1)
        spec_sum = torch.sum(spec, dim=2)
        centroid = torch.sum(freqs.unsqueeze(1) * spec_sum, dim=1) / torch.sum(spec_sum, dim=1)
        
        # Calculate spectral bandwidth
        bandwidth = torch.sqrt(torch.sum(((freqs.unsqueeze(1) - centroid.unsqueeze(0))**2) * spec_sum, dim=1) / torch.sum(spec_sum, dim=1))
        
        # Calculate spectral contrast in low, mid, and high frequency bands
        low_band = torch.mean(spec[:, :n_fft//6, :], dim=(1, 2))
        mid_band = torch.mean(spec[:, n_fft//6:n_fft//3, :], dim=(1, 2))
        high_band = torch.mean(spec[:, n_fft//3:, :], dim=(1, 2))
        
        # Calculate spectral flatness
        eps = 1e-10
        log_spec = torch.log(spec + eps)
        flatness = torch.exp(torch.mean(log_spec, dim=1)) / (torch.mean(spec, dim=1) + eps)
        flatness = torch.mean(flatness, dim=1)
        
        # Combine features
        features = torch.cat([
            rms.unsqueeze(0),
            peak.unsqueeze(0),
            centroid,
            bandwidth,
            low_band,
            mid_band,
            high_band,
            flatness
        ])
        
        # Ensure we have exactly 10 features
        if features.shape[0] < 10:
            features = torch.cat([features, torch.zeros(10 - features.shape[0])])
        elif features.shape[0] > 10:
            features = features[:10]
            
        return features.detach().cpu().numpy()
    
    def calculate_snr(self, original_audio, stego_audio):
        """Calculate Signal-to-Noise Ratio between original and stego audio"""
        noise = stego_audio - original_audio
        signal_power = torch.mean(original_audio**2)
        noise_power = torch.mean(noise**2)
        snr = 10 * torch.log10(signal_power / (noise_power + 1e-10))
        return snr.item()
    
    def calculate_bit_accuracy(self, original_message, decoded_message):
        """Calculate bit accuracy between original and decoded messages"""
        min_len = min(len(original_message), len(decoded_message))
        if min_len == 0:
            return 0.0
        
        correct_bits = sum(o == d for o, d in zip(original_message[:min_len], decoded_message[:min_len]))
        return correct_bits / min_len
    
    def apply_frequency_band_selection(self, audio, message, band_selection):
        """Apply message to specific frequency band of audio"""
        # Convert to frequency domain
        n_fft = 1024
        hop_length = 512
        window = torch.hann_window(n_fft)
        
        # Compute STFT
        stft = torch.stft(audio.squeeze(0), n_fft, hop_length, window=window, return_complex=True)
        
        # Determine frequency band boundaries
        if band_selection < 0.5:  # Low frequencies
            band_start, band_end = 0, n_fft // 6
        elif band_selection < 1.5:  # Mid frequencies
            band_start, band_end = n_fft // 6, n_fft // 3
        else:  # High frequencies
            band_start, band_end = n_fft // 3, n_fft // 2 + 1
        
        # Repeat message to match the number of frames
        message_expanded = message.repeat(1, 1, stft.shape[1])
        
        # Apply message to selected frequency band
        message_phase = torch.angle(stft[band_start:band_end])
        message_magnitude = torch.abs(stft[band_start:band_end])
        
        # Modify magnitude based on message
        alpha = 0.05  # Embedding strength
        modified_magnitude = message_magnitude * (1 + alpha * message_expanded[:, :band_end-band_start, :])
        
        # Reconstruct complex STFT
        modified_stft = stft.clone()
        modified_stft[band_start:band_end] = modified_magnitude * torch.exp(1j * message_phase)