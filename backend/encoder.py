import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.signal
from typing import Dict, Tuple, Optional

class AttentionModule(nn.Module):
    """Self-attention module for better feature learning"""
    def __init__(self, channels):
        super(AttentionModule, self).__init__()
        self.channels = channels
        self.query = nn.Conv1d(channels, channels // 8, 1)
        self.key = nn.Conv1d(channels, channels // 8, 1)
        self.value = nn.Conv1d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, channels, length = x.size()
        
        # Generate query, key, value
        q = self.query(x).view(batch_size, -1, length).permute(0, 2, 1)  # [B, L, C//8]
        k = self.key(x).view(batch_size, -1, length)  # [B, C//8, L]
        v = self.value(x).view(batch_size, -1, length)  # [B, C, L]
        
        # Attention weights
        attention = torch.bmm(q, k)  # [B, L, L]
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention
        out = torch.bmm(v, attention.permute(0, 2, 1))  # [B, C, L]
        out = self.gamma * out + x
        
        return out

class AdvancedAudioStegEncoder(nn.Module):
    """Advanced steganographic encoder with multiple embedding methods and RL integration"""
    
    def __init__(self, embedding_strength=0.01, use_attention=True):
        super(AdvancedAudioStegEncoder, self).__init__()
        self.embedding_strength = embedding_strength
        self.use_attention = use_attention
        
        # Neural network layers
        self.conv1 = nn.Conv1d(2, 64, kernel_size=15, padding=7)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=15, padding=7)
        self.conv3 = nn.Conv1d(128, 64, kernel_size=15, padding=7)
        self.conv4 = nn.Conv1d(64, 32, kernel_size=15, padding=7)
        self.conv5 = nn.Conv1d(32, 1, kernel_size=15, padding=7)
        
        # Attention modules
        if use_attention:
            self.attention1 = AttentionModule(64)
            self.attention2 = AttentionModule(128)
            
        # Activation and regularization
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.1)
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.batch_norm3 = nn.BatchNorm1d(64)
        
        # Method selection network (for RL integration)
        self.method_selector = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 5),  # 5 embedding methods
            nn.Softmax(dim=1)
        )
        
    def forward(self, audio, message, method='neural', rl_params=None):
        """Forward pass with method selection and RL parameter integration"""
        # Ensure proper tensor shapes
        audio, message = self._prepare_inputs(audio, message)
        
        if method == 'neural':
            return self._neural_embedding(audio, message, rl_params)
        elif method == 'lsb':
            return self._lsb_embedding(audio, message, rl_params)
        elif method == 'spread_spectrum':
            return self._spread_spectrum_embedding(audio, message, rl_params)
        elif method == 'echo_hiding':
            return self._echo_hiding_embedding(audio, message, rl_params)
        elif method == 'phase_coding':
            return self._phase_coding_embedding(audio, message, rl_params)
        elif method == 'adaptive':
            return self._adaptive_embedding(audio, message, rl_params)
        else:
            return self._neural_embedding(audio, message, rl_params)
    
    def _prepare_inputs(self, audio, message):
        """Prepare and validate input tensors"""
        # Ensure message has channel dimension
        if message.dim() == 2:
            message = message.unsqueeze(1)
            
        # Ensure audio has channel dimension
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
            
        # Match lengths
        if message.size(-1) != audio.size(-1):
            if message.size(-1) < audio.size(-1):
                repeat_factor = audio.size(-1) // message.size(-1) + 1
                message = message.repeat(1, 1, repeat_factor)[:, :, :audio.size(-1)]
            else:
                message = message[:, :, :audio.size(-1)]
                
        return audio, message
    
    def _neural_embedding(self, audio, message, rl_params=None):
        """Advanced neural network-based embedding"""
        # Concatenate audio and message
        x = torch.cat([audio, message], dim=1)  # [B, 2, T]
        
        # First conv block with attention
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        if self.use_attention:
            x = self.attention1(x)
        x = self.dropout(x)
        
        # Second conv block with attention
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        if self.use_attention:
            x = self.attention2(x)
        x = self.dropout(x)
        
        # Third conv block
        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Fourth conv block
        x = self.conv4(x)
        x = self.relu(x)
        
        # Final layer
        stego_delta = self.conv5(x)  # [B, 1, T]
        
        # Apply RL-controlled embedding strength
        strength = self.embedding_strength
        if rl_params and 'embedding_strength' in rl_params:
            strength = rl_params['embedding_strength']
            
        return audio + strength * stego_delta
    
    def _lsb_embedding(self, audio, message, rl_params=None):
        """Improved LSB embedding with RL parameter control"""
        # Convert to numpy for processing
        audio_np = audio.squeeze().detach().cpu().numpy()
        message_np = message.squeeze().detach().cpu().numpy()
        
        # RL-controlled parameters
        bits_per_sample = 1
        if rl_params and 'bits_per_sample' in rl_params:
            bits_per_sample = int(rl_params['bits_per_sample'])
            
        # Convert to 16-bit integers
        audio_int = (audio_np * 32767).astype(np.int16)
        
        # Convert message to binary
        message_binary = ''.join([str(int(b)) for b in message_np[:len(audio_int)]])
        
        # Embed in LSBs
        for i, bit in enumerate(message_binary):
            if i >= len(audio_int):
                break
            # Clear LSBs and set to message bit
            mask = ~((1 << bits_per_sample) - 1)
            audio_int[i] = (audio_int[i] & mask) | (int(bit) << (bits_per_sample - 1))
        
        # Convert back to tensor
        result = torch.tensor(audio_int.astype(np.float32) / 32767.0, 
                             device=audio.device, dtype=audio.dtype)
        return result.unsqueeze(0).unsqueeze(0)
    
    def _spread_spectrum_embedding(self, audio, message, rl_params=None):
        """Spread spectrum steganography implementation"""
        audio_np = audio.squeeze().detach().cpu().numpy()
        message_np = message.squeeze().detach().cpu().numpy()
        
        # RL-controlled parameters
        spreading_factor = 8
        chip_rate = 1000
        if rl_params:
            spreading_factor = int(rl_params.get('spreading_factor', 8))
            chip_rate = int(rl_params.get('chip_rate', 1000))
        
        # Generate pseudo-noise sequence
        np.random.seed(42)  # Fixed seed for reproducibility
        pn_sequence = np.random.choice([-1, 1], size=len(audio_np))
        
        # Spread the message
        spread_message = np.zeros_like(audio_np)
        for i, bit in enumerate(message_np[:len(audio_np)//spreading_factor]):
            start_idx = i * spreading_factor
            end_idx = min(start_idx + spreading_factor, len(spread_message))
            spread_message[start_idx:end_idx] = (2 * bit - 1) * pn_sequence[start_idx:end_idx]
        
        # Embed with controlled strength
        strength = 0.001
        if rl_params and 'embedding_strength' in rl_params:
            strength = rl_params['embedding_strength']
            
        stego_audio = audio_np + strength * spread_message
        
        result = torch.tensor(stego_audio, device=audio.device, dtype=audio.dtype)
        return result.unsqueeze(0).unsqueeze(0)
    
    def _echo_hiding_embedding(self, audio, message, rl_params=None):
        """Echo hiding steganography implementation"""
        audio_np = audio.squeeze().detach().cpu().numpy()
        message_np = message.squeeze().detach().cpu().numpy()
        
        # RL-controlled parameters
        delay_0 = 50  # samples for bit 0
        delay_1 = 100  # samples for bit 1
        decay = 0.5
        
        if rl_params:
            delay_0 = int(rl_params.get('echo_delay_0', 50))
            delay_1 = int(rl_params.get('echo_delay_1', 100))
            decay = rl_params.get('echo_decay', 0.5)
        
        stego_audio = audio_np.copy()
        segment_length = len(audio_np) // len(message_np[:100])  # Limit message length
        
        for i, bit in enumerate(message_np[:100]):
            start_idx = i * segment_length
            end_idx = min(start_idx + segment_length, len(audio_np))
            
            if end_idx - start_idx > max(delay_0, delay_1):
                delay = delay_1 if bit > 0.5 else delay_0
                echo_start = start_idx + delay
                echo_end = min(echo_start + (end_idx - start_idx - delay), len(stego_audio))
                
                if echo_end > echo_start:
                    stego_audio[echo_start:echo_end] += decay * audio_np[start_idx:start_idx + (echo_end - echo_start)]
        
        result = torch.tensor(stego_audio, device=audio.device, dtype=audio.dtype)
        return result.unsqueeze(0).unsqueeze(0)
    
    def _phase_coding_embedding(self, audio, message, rl_params=None):
        """Phase coding steganography implementation"""
        audio_np = audio.squeeze().detach().cpu().numpy()
        message_np = message.squeeze().detach().cpu().numpy()
        
        # Parameters
        segment_length = 1024
        if rl_params and 'segment_length' in rl_params:
            segment_length = int(rl_params['segment_length'])
        
        stego_audio = audio_np.copy()
        num_segments = len(audio_np) // segment_length
        
        for i, bit in enumerate(message_np[:num_segments]):
            start_idx = i * segment_length
            end_idx = start_idx + segment_length
            
            if end_idx <= len(audio_np):
                segment = audio_np[start_idx:end_idx]
                
                # FFT
                fft_segment = np.fft.fft(segment)
                magnitude = np.abs(fft_segment)
                phase = np.angle(fft_segment)
                
                # Modify phase based on bit
                if bit > 0.5:
                    phase[1:segment_length//2] = np.pi/2
                else:
                    phase[1:segment_length//2] = -np.pi/2
                
                # Maintain conjugate symmetry
                phase[segment_length//2+1:] = -phase[1:segment_length//2][::-1]
                
                # Reconstruct
                modified_fft = magnitude * np.exp(1j * phase)
                stego_audio[start_idx:end_idx] = np.real(np.fft.ifft(modified_fft))
        
        result = torch.tensor(stego_audio, device=audio.device, dtype=audio.dtype)
        return result.unsqueeze(0).unsqueeze(0)
    
    def _adaptive_embedding(self, audio, message, rl_params=None):
        """Adaptive embedding using RL to select best method"""
        # Use method selector network to choose embedding method
        x = torch.cat([audio, message], dim=1)
        method_probs = self.method_selector(x)
        
        # Select method based on highest probability or RL action
        if rl_params and 'selected_method' in rl_params:
            method_idx = rl_params['selected_method']
        else:
            method_idx = torch.argmax(method_probs, dim=1).item()
        
        methods = ['neural', 'lsb', 'spread_spectrum', 'echo_hiding', 'phase_coding']
        selected_method = methods[method_idx]
        
        return self.forward(audio, message, method=selected_method, rl_params=rl_params)
    
    def get_embedding_capacity(self, audio_length, method='neural'):
        """Calculate embedding capacity for different methods"""
        if method == 'neural':
            return min(audio_length // 8, 1000)  # Conservative estimate
        elif method == 'lsb':
            return audio_length  # 1 bit per sample
        elif method == 'spread_spectrum':
            return audio_length // 16  # Lower capacity due to spreading
        elif method == 'echo_hiding':
            return audio_length // 100  # Very low capacity
        elif method == 'phase_coding':
            return audio_length // 32  # Moderate capacity
        else:
            return audio_length // 16
    
    def decode(self, stego_audio, method='neural', rl_params=None):
        """Decode message from steganographic audio"""
        if isinstance(stego_audio, torch.Tensor):
            audio_np = stego_audio.detach().cpu().numpy()
        else:
            audio_np = np.array(stego_audio)
            
        if method == 'neural':
            # Use neural network decoder
            with torch.no_grad():
                audio_tensor = torch.from_numpy(audio_np).float().unsqueeze(0)
                # This would need a proper decoder implementation
                return "Neural decoding not fully implemented"
                
        elif method == 'lsb':
            # Enhanced LSB decoding
            from utils import simple_lsb_extract
            return simple_lsb_extract(audio_np)
            
        elif method == 'spread_spectrum':
            # Spread spectrum decoding
            spreading_factor = rl_params.get('spreading_factor', 8) if rl_params else 8
            return self._decode_spread_spectrum(audio_np, spreading_factor)
            
        elif method == 'echo_hiding':
            # Echo hiding decoding
            delay_0 = rl_params.get('echo_delay_0', 50) if rl_params else 50
            delay_1 = rl_params.get('echo_delay_1', 100) if rl_params else 100
            return self._decode_echo_hiding(audio_np, delay_0, delay_1)
            
        elif method == 'phase_coding':
            # Phase coding decoding
            segment_length = rl_params.get('segment_length', 1024) if rl_params else 1024
            return self._decode_phase_coding(audio_np, segment_length)
            
        else:
            return "Unknown decoding method"
    
    def _decode_spread_spectrum(self, stego_audio, spreading_factor):
        """Decode spread spectrum embedded message"""
        try:
            # Simplified spread spectrum decoding
            # In practice, this would correlate with the spreading sequence
            message_bits = []
            
            for i in range(0, len(stego_audio) - spreading_factor, spreading_factor):
                segment = stego_audio[i:i+spreading_factor]
                # Simple correlation-based detection
                bit_value = 1 if np.mean(segment) > 0 else 0
                message_bits.append(str(bit_value))
                
                # Stop at reasonable message length
                if len(message_bits) >= 800:  # 100 characters max
                    break
            
            # Convert bits to text
            message = ""
            for i in range(0, len(message_bits) - 7, 8):
                byte_bits = ''.join(message_bits[i:i+8])
                if len(byte_bits) == 8:
                    char_code = int(byte_bits, 2)
                    if 32 <= char_code <= 126:  # Printable ASCII
                        message += chr(char_code)
                    elif char_code == 0:  # End marker
                        break
            
            return message if message else "No message found"
            
        except Exception as e:
            return f"Decoding error: {str(e)}"
    
    def _decode_echo_hiding(self, stego_audio, delay_0, delay_1):
        """Decode echo hiding embedded message"""
        try:
            message_bits = []
            window_size = max(delay_0, delay_1) + 50
            
            for i in range(0, len(stego_audio) - window_size, window_size):
                segment = stego_audio[i:i+window_size]
                
                # Check for echo at delay_0 (bit 0) vs delay_1 (bit 1)
                autocorr_0 = np.corrcoef(segment[:-delay_0], segment[delay_0:])[0, 1] if len(segment) > delay_0 else 0
                autocorr_1 = np.corrcoef(segment[:-delay_1], segment[delay_1:])[0, 1] if len(segment) > delay_1 else 0
                
                # Determine bit based on stronger correlation
                bit_value = '1' if autocorr_1 > autocorr_0 else '0'
                message_bits.append(bit_value)
                
                if len(message_bits) >= 800:  # Limit message length
                    break
            
            # Convert bits to text
            message = ""
            for i in range(0, len(message_bits) - 7, 8):
                byte_bits = ''.join(message_bits[i:i+8])
                if len(byte_bits) == 8:
                    char_code = int(byte_bits, 2)
                    if 32 <= char_code <= 126:  # Printable ASCII
                        message += chr(char_code)
                    elif char_code == 0:  # End marker
                        break
            
            return message if message else "No message found"
            
        except Exception as e:
            return f"Decoding error: {str(e)}"
    
    def _decode_phase_coding(self, stego_audio, segment_length):
        """Decode phase coding embedded message"""
        try:
            message_bits = []
            
            for i in range(0, len(stego_audio) - segment_length, segment_length):
                segment = stego_audio[i:i+segment_length]
                
                # Compute FFT and extract phase
                fft = np.fft.fft(segment)
                phases = np.angle(fft)
                
                # Simple phase-based bit extraction
                # Check phase difference in middle frequencies
                mid_freq = len(phases) // 4
                phase_diff = phases[mid_freq] - phases[mid_freq + 1]
                
                bit_value = '1' if phase_diff > 0 else '0'
                message_bits.append(bit_value)
                
                if len(message_bits) >= 800:  # Limit message length
                    break
            
            # Convert bits to text
            message = ""
            for i in range(0, len(message_bits) - 7, 8):
                byte_bits = ''.join(message_bits[i:i+8])
                if len(byte_bits) == 8:
                    char_code = int(byte_bits, 2)
                    if 32 <= char_code <= 126:  # Printable ASCII
                        message += chr(char_code)
                    elif char_code == 0:  # End marker
                        break
            
            return message if message else "No message found"
            
        except Exception as e:
            return f"Decoding error: {str(e)}"

# Backward compatibility
class AudioStegEncoder(AdvancedAudioStegEncoder):
    """Backward compatible encoder class"""
    def __init__(self):
        super(AudioStegEncoder, self).__init__(embedding_strength=0.01, use_attention=False)
        
    def forward(self, audio, message):
        return super().forward(audio, message, method='neural')
