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
            shape=(47,),  # Expanded feature vector (reduced from 50 due to spectral contrast bands)
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
        try:
            # Ensure audio is finite
            if not np.isfinite(audio).all():
                return np.zeros(50, dtype=np.float32)
            
            features = []
            
            # Time domain features with safety checks
            try:
                mean_val = np.mean(audio)
                std_val = np.std(audio)
                max_abs = np.max(np.abs(audio))
                rms = np.sqrt(np.mean(audio**2))
                zcr = len(np.where(np.diff(np.sign(audio)))[0]) / max(len(audio), 1)
                
                time_features = [mean_val, std_val, max_abs, rms, zcr]
                # Replace non-finite values with 0
                time_features = [f if np.isfinite(f) else 0.0 for f in time_features]
                features.extend(time_features)
            except Exception:
                features.extend([0.0] * 5)
            
            # Frequency domain features using FFT
            try:
                fft = np.fft.fft(audio)
                magnitude = np.abs(fft[:len(fft)//2])
                
                # Spectral features
                freqs = np.fft.fftfreq(len(audio), 1/self.sample_rate)[:len(fft)//2]
                
                mag_sum = np.sum(magnitude)
                if mag_sum > 1e-8:
                    spectral_centroid = np.sum(freqs * magnitude) / mag_sum
                    spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * magnitude) / mag_sum)
                    cumsum_mag = np.cumsum(magnitude)
                    rolloff_idx = np.where(cumsum_mag >= 0.85 * mag_sum)[0]
                    spectral_rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0.0
                else:
                    spectral_centroid = spectral_bandwidth = spectral_rolloff = 0.0
                
                spectral_features = [spectral_centroid, spectral_bandwidth, spectral_rolloff]
                spectral_features = [f if np.isfinite(f) else 0.0 for f in spectral_features]
                features.extend(spectral_features)
            except Exception:
                features.extend([0.0] * 3)
            
            # MFCC features (first 13 coefficients)
            try:
                mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
                mfcc_means = np.mean(mfccs, axis=1)
                mfcc_means = [f if np.isfinite(f) else 0.0 for f in mfcc_means]
                features.extend(mfcc_means)
            except Exception:
                features.extend([0.0] * 13)
            
            # Spectral contrast (4 bands - reduced to avoid Nyquist frequency issues)
            try:
                contrast = librosa.feature.spectral_contrast(y=audio, sr=self.sample_rate, n_bands=4)
                contrast_means = np.mean(contrast, axis=1)
                contrast_means = [f if np.isfinite(f) else 0.0 for f in contrast_means]
                features.extend(contrast_means)
            except Exception:
                features.extend([0.0] * 5)  # n_bands + 1
            
            # Chroma features (12 bins)
            try:
                chroma = librosa.feature.chroma_stft(y=audio, sr=self.sample_rate)
                chroma_means = np.mean(chroma, axis=1)
                chroma_means = [f if np.isfinite(f) else 0.0 for f in chroma_means]
                features.extend(chroma_means)
            except Exception:
                features.extend([0.0] * 12)
            
            # Tonnetz features (6 dimensions)
            try:
                tonnetz = librosa.feature.tonnetz(y=audio, sr=self.sample_rate)
                tonnetz_means = np.mean(tonnetz, axis=1)
                tonnetz_means = [f if np.isfinite(f) else 0.0 for f in tonnetz_means]
                features.extend(tonnetz_means)
            except Exception:
                features.extend([0.0] * 6)
            
            # Pad or truncate to exactly 50 features
            features = np.array(features[:50])
            if len(features) < 50:
                features = np.pad(features, (0, 50 - len(features)))
            
            # Final check for finite values
            features = np.where(np.isfinite(features), features, 0.0)
            
            return features.astype(np.float32)
        except Exception:
            return np.zeros(50, dtype=np.float32)
    
    def reset(self):
        """Reset environment to initial state"""
        self.current_step = 0
        self.current_audio = self.original_audio.copy()
        return self.extract_audio_features(self.current_audio)
    
    def step(self, action):
        """Execute steganography action"""
        self.current_step += 1
        
        # Handle both integer actions and array actions
        if isinstance(action, (int, np.integer)):
            # Convert integer action to method selection with default parameters
            method_idx = action % len(self.methods)
            embedding_strength = 0.1  # Default strength
            freq_start = 0.1  # Default frequency start
            freq_end = 0.9    # Default frequency end
            temporal_pos = 0.5  # Default temporal position
        else:
            # Parse action parameters from array
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
    
    def get_current_audio(self):
        """Get the current modified audio"""
        return self.current_audio
    
    def decode_message(self, num_bits=None):
        """Decode message from current audio using LSB extraction"""
        try:
            # Ensure current audio is finite
            if not np.isfinite(self.current_audio).all():
                return '0' * (num_bits or 100)
            
            # Convert audio to int16 for LSB extraction
            audio_clipped = np.clip(self.current_audio, -1.0, 1.0)
            audio_int = (audio_clipped * 32767).astype(np.int16)
            
            # Extract LSBs
            extracted_bits = []
            max_bits = num_bits or min(len(audio_int), 1000)  # Default to 1000 bits max
            
            for i in range(min(len(audio_int), max_bits)):
                lsb = audio_int[i] & 1
                extracted_bits.append(str(lsb))
            
            return ''.join(extracted_bits)
        except Exception:
            return '0' * (num_bits or 100)
    
    def calculate_snr(self, original, modified):
        """Calculate Signal-to-Noise Ratio"""
        try:
            # Ensure inputs are finite
            if not (np.isfinite(original).all() and np.isfinite(modified).all()):
                return 0.0
            
            noise = modified - original
            signal_power = np.mean(original ** 2)
            noise_power = np.mean(noise ** 2)
            
            # Handle edge cases
            if signal_power < 1e-10 or noise_power < 1e-10:
                return 100.0  # Very high SNR
            
            snr = 10 * np.log10(signal_power / noise_power)
            
            # Ensure result is finite
            if not np.isfinite(snr):
                return 0.0
                
            return float(snr)
        except Exception:
            return 0.0
    
    def calculate_bit_accuracy(self, original_bits, decoded_bits):
        """Calculate bit accuracy between original and decoded messages"""
        min_len = min(len(original_bits), len(decoded_bits))
        if min_len == 0:
            return 0.0
        matches = sum(1 for i in range(min_len) if original_bits[i] == decoded_bits[i])
        return matches / min_len
    
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
        try:
            # Ensure audio is finite and within valid range
            if not np.isfinite(audio).all():
                return audio.copy()
            
            # Clip audio to valid range before conversion
            audio_clipped = np.clip(audio, -1.0, 1.0)
            audio_int = (audio_clipped * 32767).astype(np.int16)
            original_int = audio_int.copy()
            
            # Embed message bits in LSB
            for i, bit in enumerate(message_bits):
                if i >= len(audio_int):
                    break
                # Clear LSB first, then set to message bit
                audio_int[i] = (audio_int[i] & ~1) | int(bit)
            
            # Convert back to float
            modified = audio_int.astype(np.float32) / 32767
            
            # Apply strength control - blend between original and fully modified
            if strength >= 1.0:
                # Full strength - use the modified audio directly
                result = modified
            else:
                # Partial strength - blend original and modified
                result = audio + strength * (modified - audio)
            
            # Ensure all values are finite and in valid range
            if not np.isfinite(result).all():
                return audio.copy()
            
            result = np.clip(result, -1.0, 1.0)
            return result
        except Exception:
            return audio.copy()
    
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
        try:
            # Base reward for successful embedding
            base_reward = 0.5
            
            # Audio quality (SNR) - improved calculation
            if not (np.isfinite(self.current_audio).all() and np.isfinite(self.original_audio).all()):
                return 0.0
                
            noise = self.current_audio - self.original_audio
            signal_power = np.mean(self.original_audio ** 2)
            noise_power = np.mean(noise ** 2)
            
            if noise_power < 1e-10 or signal_power < 1e-10:  # Very little noise - good!
                snr_reward = 1.0
            else:
                snr = 10 * np.log10(signal_power / noise_power)
                if not np.isfinite(snr):
                    snr_reward = 0.0
                else:
                    # More generous SNR reward - target 20dB+
                    snr_reward = max(0, min(1.0, (snr + 10) / 30.0))  # Maps -10dB to 0, 20dB to 1
        
            # Imperceptibility (perceptual distance)
            perceptual_reward = self.calculate_perceptual_quality()
            
            # Capacity utilization - more generous
            capacity_ratio = len(self.message_bits) / max(len(self.current_audio) * 0.001, 1)
            capacity_reward = min(capacity_ratio, 1.0)
            
            # Robustness to common attacks
            robustness_reward = self.calculate_robustness()
            
            # Security (statistical undetectability)
            security_reward = 1.0 - self.calculate_statistical_detectability()
            
            # Ensure all rewards are finite
            rewards = [snr_reward, perceptual_reward, capacity_reward, robustness_reward, security_reward]
            if not all(np.isfinite(r) for r in rewards):
                return base_reward
            
            # Weighted combination with base reward
            total_reward = base_reward + (
                0.25 * snr_reward +
                0.2 * perceptual_reward +
                0.15 * capacity_reward +
                0.2 * robustness_reward +
                0.2 * security_reward
            )
            
            return float(total_reward) if np.isfinite(total_reward) else base_reward
        except Exception:
            return 0.5  # Return base reward on any error
    
    def calculate_perceptual_quality(self):
        """Calculate perceptual audio quality using spectral features"""
        try:
            # Compare spectral features between original and stego audio
            orig_features = self.extract_audio_features(self.original_audio)
            stego_features = self.extract_audio_features(self.current_audio)
            
            # Ensure features are finite
            if not (np.isfinite(orig_features).all() and np.isfinite(stego_features).all()):
                return 0.3  # Return minimum quality
            
            # Calculate feature distance with more generous scaling
            feature_distance = np.linalg.norm(orig_features - stego_features)
            
            # Ensure distance is finite
            if not np.isfinite(feature_distance):
                return 0.3
            
            # More generous perceptual quality - smaller changes get higher rewards
            perceptual_quality = np.exp(-feature_distance / 50.0)  # More generous scaling
            
            # Ensure result is finite
            if not np.isfinite(perceptual_quality):
                return 0.3
            
            # Ensure minimum quality score
            perceptual_quality = max(0.3, perceptual_quality)  # Minimum 0.3 quality
            
            return float(perceptual_quality)
        except Exception:
            return 0.3
    
    def calculate_robustness(self):
        """Test robustness against common signal processing attacks"""
        try:
            robustness_scores = []
            
            # Ensure current audio is finite
            if not np.isfinite(self.current_audio).all():
                return 0.0
            
            # Test against compression
            compressed = self.simulate_mp3_compression(self.current_audio)
            if np.isfinite(compressed).all():
                score = self.test_message_recovery(compressed)
                if np.isfinite(score):
                    robustness_scores.append(score)
            
            # Test against noise
            noisy = self.current_audio + np.random.normal(0, 0.001, len(self.current_audio))
            if np.isfinite(noisy).all():
                score = self.test_message_recovery(noisy)
                if np.isfinite(score):
                    robustness_scores.append(score)
            
            # Test against resampling
            try:
                resampled = scipy.signal.resample(self.current_audio, int(len(self.current_audio) * 0.9))
                resampled = scipy.signal.resample(resampled, len(self.current_audio))
                if np.isfinite(resampled).all():
                    score = self.test_message_recovery(resampled)
                    if np.isfinite(score):
                        robustness_scores.append(score)
            except Exception:
                pass  # Skip resampling test if it fails
            
            if len(robustness_scores) == 0:
                return 0.0
            
            result = np.mean(robustness_scores)
            return float(result) if np.isfinite(result) else 0.0
        except Exception:
            return 0.0
    
    def simulate_mp3_compression(self, audio):
        """Simulate MP3 compression effects"""
        # Simple simulation using lowpass filtering and quantization
        # Real implementation would use actual MP3 codec
        from scipy.signal import butter, filtfilt
        
        # Lowpass filter to simulate frequency cutoff
        nyquist = self.sample_rate / 2
        cutoff = min(11025, nyquist * 0.95)  # Ensure cutoff < nyquist
        normalized_cutoff = cutoff / nyquist
        b, a = butter(5, normalized_cutoff, btype='low')
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
        try:
            # Ensure audio is finite and in valid range
            orig_audio = np.clip(self.original_audio, -1.0, 1.0)
            curr_audio = np.clip(self.current_audio, -1.0, 1.0)
            
            # Replace any non-finite values
            orig_audio = np.where(np.isfinite(orig_audio), orig_audio, 0.0)
            curr_audio = np.where(np.isfinite(curr_audio), curr_audio, 0.0)
            
            # Chi-square test for LSB randomness
            orig_lsb = (orig_audio * 32767).astype(np.int16) & 1
            stego_lsb = (curr_audio * 32767).astype(np.int16) & 1
            
            # Count bit changes
            changes = np.sum(orig_lsb != stego_lsb)
            total_samples = len(orig_lsb)
            
            if total_samples == 0:
                return 0.0
            
            # Simple detectability metric
            detectability = changes / total_samples
            
            return min(detectability * 10, 1.0)  # Scale to [0,1]
        except Exception:
            return 0.0
    
    def get_info(self):
        """Return additional information about current state"""
        return {
            'snr': self.calculate_current_snr(),
            'message_length': self.message_length,
            'audio_length': len(self.current_audio),
            'step': self.current_step
        }
    
    def calculate_current_snr(self):
        """Calculate current SNR between original and current audio"""
        return self.calculate_snr(self.original_audio, self.current_audio)