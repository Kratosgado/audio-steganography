import numpy as np
import torch
import torch.nn as nn
import gym
from gym import spaces
import librosa
import scipy.signal

class AudioStegEnvironment(gym.Env):
    """Enhanced environment for audio steganography with more sophisticated features"""
    
    def __init__(self, audio_path=None, message=None, max_steps=50, sample_rate=16000):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.max_steps = max_steps
        self.current_step = 0
        
        # Enhanced Action Space: Continuous control over steganography parameters
        # [embedding_strength, frequency_start, frequency_end, temporal_position, method_type]
        self.action_space = spaces.Box(
            low=np.array([0.001, 0.0, 0.1, 0.0, 0.0]),      # Min values
            high=np.array([0.1, 0.4, 1.0, 1.0, 1.0]),       # Max values
            dtype=np.float32
        )
        
        # Enhanced State Space: Comprehensive audio features
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(50,),  # Expanded feature vector
            dtype=np.float32
        )
        
        # Steganography methods
        self.methods = ['lsb', 'dct', 'dwt', 'spectral', 'phase']
        
        # Load audio and message
        self.load_data(audio_path, message)
        
    def load_data(self, audio_path, message):
        """Load and preprocess audio and message data"""
        if audio_path:
            try:
                self.original_audio, _ = librosa.load(audio_path, sr=self.sample_rate)
            except:
                self.original_audio = np.random.randn(self.sample_rate * 2)  # 2 seconds
        else:
            self.original_audio = np.random.randn(self.sample_rate * 2)
            
        self.current_audio = self.original_audio.copy()
        
        # Convert message to binary
        if message:
            self.message_bits = ''.join(format(ord(char), '08b') for char in message)
        else:
            self.message_bits = '0110100001100101011011000110110001101111'  # "hello"
            
        self.message_length = len(self.message_bits)
        
    def extract_audio_features(self, audio):
        """Extract comprehensive audio features for better state representation"""
        features = []
        
        # Time domain features
        features.extend([
            np.mean(audio),
            np.std(audio),
            np.max(np.abs(audio)),
            np.sqrt(np.mean(audio**2)),  # RMS
            len(np.where(np.diff(np.sign(audio)))[0]) / len(audio),  # Zero crossing rate
        ])
        
        # Frequency domain features using FFT
        fft = np.fft.fft(audio)
        magnitude = np.abs(fft[:len(fft)//2])
        
        # Spectral features
        freqs = np.fft.fftfreq(len(audio), 1/self.sample_rate)[:len(fft)//2]
        spectral_centroid = np.sum(freqs * magnitude) / (np.sum(magnitude) + 1e-8)
        spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * magnitude) / (np.sum(magnitude) + 1e-8))
        spectral_rolloff = freqs[np.where(np.cumsum(magnitude) >= 0.85 * np.sum(magnitude))[0][0]]
        
        features.extend([spectral_centroid, spectral_bandwidth, spectral_rolloff])
        
        # MFCC features (first 13 coefficients)
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
        features.extend(np.mean(mfccs, axis=1))
        
        # Spectral contrast (7 bands)
        contrast = librosa.feature.spectral_contrast(y=audio, sr=self.sample_rate, n_bands=7)
        features.extend(np.mean(contrast, axis=1))
        
        # Chroma features (12 bins)
        chroma = librosa.feature.chroma_stft(y=audio, sr=self.sample_rate)
        features.extend(np.mean(chroma, axis=1))
        
        # Tonnetz features (6 dimensions)
        tonnetz = librosa.feature.tonnetz(y=audio, sr=self.sample_rate)
        features.extend(np.mean(tonnetz, axis=1))
        
        # Pad or truncate to exactly 50 features
        features = np.array(features[:50])
        if len(features) < 50:
            features = np.pad(features, (0, 50 - len(features)))
            
        return features.astype(np.float32)
    
    def reset(self):
        """Reset environment to initial state"""
        self.current_step = 0
        self.current_audio = self.original_audio.copy()
        return self.extract_audio_features(self.current_audio)
    
    def step(self, action):
        """Execute steganography action"""
        self.current_step += 1
        
        # Parse action parameters
        embedding_strength = action[0]
        freq_start = action[1]
        freq_end = max(action[2], freq_start + 0.1)  # Ensure valid range
        temporal_pos = action[3]
        method_idx = int(action[4] * len(self.methods))
        method = self.methods[min(method_idx, len(self.methods) - 1)]
        
        # Apply steganography
        try:
            self.current_audio = self.apply_steganography(
                self.current_audio, self.message_bits, 
                method, embedding_strength, freq_start, freq_end, temporal_pos
            )
        except Exception as e:
            # Fallback to simple modification
            self.current_audio = self.original_audio + np.random.normal(0, 0.001, len(self.original_audio))
        
        # Calculate reward
        reward = self.calculate_comprehensive_reward()
        
        # Check termination
        done = self.current_step >= self.max_steps
        
        # Get next state
        next_state = self.extract_audio_features(self.current_audio)
        
        return next_state, reward, done, self.get_info()
    
    def apply_steganography(self, audio, message_bits, method, strength, freq_start, freq_end, temporal_pos):
        """Apply different steganography methods"""
        if method == 'lsb':
            return self.lsb_embedding(audio, message_bits, strength)
        elif method == 'dct':
            return self.dct_embedding(audio, message_bits, strength, freq_start, freq_end)
        elif method == 'dwt':
            return self.dwt_embedding(audio, message_bits, strength)
        elif method == 'spectral':
            return self.spectral_embedding(audio, message_bits, strength, freq_start, freq_end)
        elif method == 'phase':
            return self.phase_embedding(audio, message_bits, strength, freq_start, freq_end)
        else:
            return self.lsb_embedding(audio, message_bits, strength)
    
    # ========== Embedding Algorithms ============
    def lsb_embedding(self, audio, message_bits, strength):
        """LSB steganography with controlled strength"""
        audio_int = (audio * 32767).astype(np.int16)
        
        for i, bit in enumerate(message_bits):
            if i >= len(audio_int):
                break
            if int(bit):
                audio_int[i] = audio_int[i] | 1
            else:
                audio_int[i] = audio_int[i] & ~1
        
        # Apply strength control
        modified = audio_int.astype(np.float32) / 32767
        return audio + strength * (modified - audio)
    
    def dct_embedding(self, audio, message_bits, strength, freq_start, freq_end):
        """DCT-based steganography"""
        # Apply DCT
        dct_coeffs = scipy.fftpack.dct(audio)
        
        # Select frequency band
        start_idx = int(freq_start * len(dct_coeffs))
        end_idx = int(freq_end * len(dct_coeffs))
        
        # Embed message in selected coefficients
        for i, bit in enumerate(message_bits):
            coeff_idx = start_idx + (i % (end_idx - start_idx))
            if coeff_idx >= len(dct_coeffs):
                break
                
            if int(bit):
                dct_coeffs[coeff_idx] += strength * abs(dct_coeffs[coeff_idx])
            else:
                dct_coeffs[coeff_idx] -= strength * abs(dct_coeffs[coeff_idx])
        
        # Inverse DCT
        return scipy.fftpack.idct(dct_coeffs)
    
    def spectral_embedding(self, audio, message_bits, strength, freq_start, freq_end):
        """Spectral domain steganography"""
        # FFT
        fft = np.fft.fft(audio)
        
        # Select frequency band
        start_idx = int(freq_start * len(fft) // 2)
        end_idx = int(freq_end * len(fft) // 2)
        
        # Embed in magnitude spectrum
        for i, bit in enumerate(message_bits):
            freq_idx = start_idx + (i % (end_idx - start_idx))
            if freq_idx >= len(fft) // 2:
                break
                
            if int(bit):
                fft[freq_idx] *= (1 + strength)
                fft[-(freq_idx + 1)] *= (1 + strength)  # Maintain symmetry
            else:
                fft[freq_idx] *= (1 - strength)
                fft[-(freq_idx + 1)] *= (1 - strength)
        
        # IFFT
        return np.real(np.fft.ifft(fft))
    
    def phase_embedding(self, audio, message_bits, strength, freq_start, freq_end):
        """Phase-based steganography"""
        # STFT for phase manipulation
        f, t, stft = scipy.signal.stft(audio, nperseg=1024)
        
        # Select frequency range
        start_freq = int(freq_start * len(f))
        end_freq = int(freq_end * len(f))
        
        # Embed in phase
        phase = np.angle(stft)
        for i, bit in enumerate(message_bits):
            freq_idx = start_freq + (i % (end_freq - start_freq))
            time_idx = i % stft.shape[1]
            
            if freq_idx >= len(f) or time_idx >= stft.shape[1]:
                break
                
            if int(bit):
                phase[freq_idx, time_idx] += strength * np.pi
            else:
                phase[freq_idx, time_idx] -= strength * np.pi
        
        # Reconstruct with modified phase
        magnitude = np.abs(stft)
        modified_stft = magnitude * np.exp(1j * phase)
        
        _, reconstructed = scipy.signal.istft(modified_stft)
        return reconstructed[:len(audio)]  # Trim to original length
    
    def dwt_embedding(self, audio, message_bits, strength):
        """Discrete Wavelet Transform steganography"""
        try:
            import pywt
            coeffs = pywt.wavedec(audio, 'db4', level=4)
            
            # Embed in detail coefficients
            for i, bit in enumerate(message_bits):
                coeff_level = (i % 3) + 1  # Use levels 1-3
                if coeff_level < len(coeffs):
                    coeff_idx = i % len(coeffs[coeff_level])
                    if int(bit):
                        coeffs[coeff_level][coeff_idx] += strength * abs(coeffs[coeff_level][coeff_idx])
                    else:
                        coeffs[coeff_level][coeff_idx] -= strength * abs(coeffs[coeff_level][coeff_idx])
            
            return pywt.waverec(coeffs, 'db4')
        except ImportError:
            # Fallback to DCT if pywt not available
            return self.dct_embedding(audio, message_bits, strength, 0.1, 0.9)
    
    def calculate_comprehensive_reward(self):
        """Calculate comprehensive reward based on multiple criteria"""
        # Audio quality (SNR)
        noise = self.current_audio - self.original_audio
        signal_power = np.mean(self.original_audio ** 2)
        noise_power = np.mean(noise ** 2)
        snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
        snr_reward = np.tanh(snr / 40.0)  # Normalize to [0,1]
        
        # Imperceptibility (perceptual distance)
        perceptual_reward = self.calculate_perceptual_quality()
        
        # Capacity utilization
        capacity_reward = min(len(self.message_bits) / (len(self.current_audio) * 0.01), 1.0)
        
        # Robustness to common attacks
        robustness_reward = self.calculate_robustness()
        
        # Security (statistical undetectability)
        security_reward = 1.0 - self.calculate_statistical_detectability()
        
        # Weighted combination
        total_reward = (
            0.3 * snr_reward +
            0.25 * perceptual_reward +
            0.15 * capacity_reward +
            0.15 * robustness_reward +
            0.15 * security_reward
        )
        
        return total_reward
    
    def calculate_perceptual_quality(self):
        """Calculate perceptual audio quality using spectral features"""
        # Compare spectral features between original and stego audio
        orig_features = self.extract_audio_features(self.original_audio)
        stego_features = self.extract_audio_features(self.current_audio)
        
        # Calculate feature distance
        feature_distance = np.linalg.norm(orig_features - stego_features)
        perceptual_quality = np.exp(-feature_distance / 10.0)  # Exponential decay
        
        return perceptual_quality
    
    def calculate_robustness(self):
        """Test robustness against common signal processing attacks"""
        robustness_scores = []
        
        # Test against compression
        compressed = self.simulate_mp3_compression(self.current_audio)
        robustness_scores.append(self.test_message_recovery(compressed))
        
        # Test against noise
        noisy = self.current_audio + np.random.normal(0, 0.001, len(self.current_audio))
        robustness_scores.append(self.test_message_recovery(noisy))
        
        # Test against resampling
        resampled = scipy.signal.resample(self.current_audio, int(len(self.current_audio) * 0.9))
        resampled = scipy.signal.resample(resampled, len(self.current_audio))
        robustness_scores.append(self.test_message_recovery(resampled))
        
        return np.mean(robustness_scores)
    
    def simulate_mp3_compression(self, audio):
        """Simulate MP3 compression effects"""
        # Simple simulation using lowpass filtering and quantization
        # Real implementation would use actual MP3 codec
        from scipy.signal import butter, filtfilt
        
        # Lowpass filter to simulate frequency cutoff
        nyquist = self.sample_rate / 2
        cutoff = 11025  # MP3 cutoff frequency
        b, a = butter(5, cutoff / nyquist, btype='low')
        filtered = filtfilt(b, a, audio)
        
        # Quantization simulation
        quantized = np.round(filtered * 1024) / 1024
        
        return quantized
    
    def test_message_recovery(self, audio):
        """Test if message can be recovered from audio"""
        # Simplified recovery test - would need actual decoder
        # For now, check if audio hasn't changed too much
        correlation = np.corrcoef(self.current_audio, audio)[0, 1]
        return max(0, correlation)
    
    def calculate_statistical_detectability(self):
        """Calculate statistical detectability using various tests"""
        # Chi-square test for LSB randomness
        orig_lsb = (self.original_audio * 32767).astype(np.int16) & 1
        stego_lsb = (self.current_audio * 32767).astype(np.int16) & 1
        
        # Count bit changes
        changes = np.sum(orig_lsb != stego_lsb)
        total_samples = len(orig_lsb)
        
        # Simple detectability metric
        detectability = changes / total_samples
        
        return min(detectability * 10, 1.0)  # Scale to [0,1]
    
    def get_info(self):
        """Return additional information about current state"""
        return {
            'snr': self.calculate_snr(),
            'message_length': self.message_length,
            'audio_length': len(self.current_audio),
            'step': self.current_step
        }
    
    def calculate_snr(self):
        """Calculate current SNR"""
        noise = self.current_audio - self.original_audio
        signal_power = np.mean(self.original_audio ** 2)
        noise_power = np.mean(noise ** 2)
        return 10 * np.log10(signal_power / (noise_power + 1e-10))