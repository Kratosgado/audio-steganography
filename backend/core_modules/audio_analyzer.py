import numpy as np
import librosa
from typing import Tuple, Dict, Any


class AudioAnalyzer:
    """Analyzes audio features and calculates optimal action values for steganography"""
    
    def __init__(self):
        self.action_ranges = {
            "carrier_freq": (5000, 15000),
            "chip_rate": (50, 200),
            "snr": (10, 30),
        }
    
    def extract_audio_features(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract key audio features"""
        features = {}
        
        # Spectral features
        features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
        features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))
        features['spectral_flatness'] = np.mean(librosa.feature.spectral_flatness(y=audio))
        
        # Temporal features
        features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(audio))
        features['rms_energy'] = np.sqrt(np.mean(audio**2))
        
        # Frequency domain analysis
        fft = np.fft.fft(audio)
        magnitude_spectrum = np.abs(fft)
        features['dominant_freq'] = np.argmax(magnitude_spectrum[:len(magnitude_spectrum)//2]) * sr / len(audio)
        
        # Energy distribution
        low_band = np.sum(magnitude_spectrum[:len(magnitude_spectrum)//4])
        mid_band = np.sum(magnitude_spectrum[len(magnitude_spectrum)//4:len(magnitude_spectrum)//2])
        high_band = np.sum(magnitude_spectrum[len(magnitude_spectrum)//2:])
        total_energy = low_band + mid_band + high_band
        
        features['low_band_ratio'] = low_band / total_energy
        features['mid_band_ratio'] = mid_band / total_energy
        features['high_band_ratio'] = high_band / total_energy
        
        return features
    
    def calculate_optimal_actions(self, audio: np.ndarray, sr: int, message_length: int = None) -> Tuple[float, float, float]:
        """Calculate optimal action values based on audio features"""
        features = self.extract_audio_features(audio, sr)
        
        # Calculate carrier frequency
        carrier_freq = self._calculate_carrier_frequency(features, sr)
        
        # Calculate chip rate
        chip_rate = self._calculate_chip_rate(features, message_length)
        
        # Calculate SNR
        snr_db = self._calculate_snr(features, message_length)
        
        # Normalize to [-1, 1] range
        carrier_freq_norm = self._normalize_to_range(carrier_freq, self.action_ranges["carrier_freq"])
        chip_rate_norm = self._normalize_to_range(chip_rate, self.action_ranges["chip_rate"])
        snr_norm = self._normalize_to_range(snr_db, self.action_ranges["snr"])
        
        print(f"Audio Analysis - Carrier: {carrier_freq:.0f}Hz, Chip Rate: {chip_rate:.0f}, SNR: {snr_db:.1f}dB")
        
        return carrier_freq_norm, chip_rate_norm, snr_norm
    
    def _calculate_carrier_frequency(self, features: Dict[str, float], sr: int) -> float:
        """Calculate optimal carrier frequency"""
        spectral_centroid = features['spectral_centroid']
        dominant_freq = features['dominant_freq']
        high_band_ratio = features['high_band_ratio']
        
        # Base carrier frequency on spectral centroid
        base_carrier = spectral_centroid * 2.5
        
        # Adjust based on dominant frequency
        if dominant_freq > 0:
            dominant_harmonics = [dominant_freq * i for i in range(1, 6)]
            min_distance = min(abs(base_carrier - h) for h in dominant_harmonics)
            if min_distance < 1000:
                base_carrier += 1000
        
        # Adjust based on high-frequency content
        if high_band_ratio > 0.3:
            base_carrier *= 1.2
        elif high_band_ratio < 0.1:
            base_carrier *= 0.8
        
        return np.clip(base_carrier, self.action_ranges["carrier_freq"][0], self.action_ranges["carrier_freq"][1])
    
    def _calculate_chip_rate(self, features: Dict[str, float], message_length: int = None) -> float:
        """Calculate optimal chip rate"""
        zero_crossing_rate = features['zero_crossing_rate']
        spectral_bandwidth = features['spectral_bandwidth']
        rms_energy = features['rms_energy']
        
        base_chip_rate = 50 + (zero_crossing_rate * 1000)
        
        if spectral_bandwidth > 2000:
            base_chip_rate *= 1.3
        elif spectral_bandwidth < 500:
            base_chip_rate *= 0.7
        
        if rms_energy > 0.1:
            base_chip_rate *= 1.2
        elif rms_energy < 0.01:
            base_chip_rate *= 0.8
        
        if message_length and message_length > 100:
            base_chip_rate *= 1.1
        
        return np.clip(base_chip_rate, self.action_ranges["chip_rate"][0], self.action_ranges["chip_rate"][1])
    
    def _calculate_snr(self, features: Dict[str, float], message_length: int = None) -> float:
        """Calculate optimal SNR"""
        rms_energy = features['rms_energy']
        spectral_flatness = features['spectral_flatness']
        
        base_snr = 15 + (rms_energy * 100)
        
        if spectral_flatness > 0.5:
            base_snr += 5
        elif spectral_flatness < 0.1:
            base_snr -= 3
        
        if message_length and message_length > 100:
            base_snr += 3
        
        return np.clip(base_snr, self.action_ranges["snr"][0], self.action_ranges["snr"][1])
    
    def _normalize_to_range(self, value: float, range_tuple: Tuple[float, float]) -> float:
        """Normalize a value from its range to [-1, 1]"""
        low, high = range_tuple
        normalized = 2 * (value - low) / (high - low) - 1
        return np.clip(normalized, -1, 1)
    
    def get_embedding_capacity_estimate(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Estimate embedding capacity"""
        duration = len(audio) / sr
        max_bits_per_second = 100
        theoretical_capacity = int(duration * max_bits_per_second / 8)
        practical_capacity = int(theoretical_capacity * 0.5)
        
        return {
            'practical_capacity_chars': practical_capacity,
            'duration_seconds': duration,
            'recommended_max_message_length': practical_capacity
        } 