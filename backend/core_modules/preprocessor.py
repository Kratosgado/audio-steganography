import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as wf

from core_modules.config import cfg

NORM_NUM = 32768


class AudioPreprocessor:
    """Handles audio loading, MDCT, and inverse MDCT."""

    def __init__(
        self,
        audio_path=None,
        audio_data=None,
        frame_size=cfg.FRAME_SIZE,
        hop_length=cfg.HOP_LENGTH,
        sr=cfg.SAMPLE_RATE,
    ):
        if audio_path:
            self.audio, self.sr = librosa.load(audio_path, sr=sr, mono=True)
        elif audio_data is not None:
            self.audio = audio_data
            self.sr = sr
        else:
            raise ValueError("Either audio_path or audio_data must be provided")

        # Normalize audio
        self.audio = self.audio / np.max(np.abs(self.audio))
        self.frame_size = frame_size
        self.hop_length = hop_length
        # return self.audio, self.sr

    def load_audio(self, path):
        """Load WAV audio file"""
        audio, _ = librosa.load(path, sr=self.sr)
        return audio

    def stft(self, audio):
        """Compute Short-Time Fourier Transform (STFT)"""
        return librosa.stft(audio, n_fft=self.frame_size, hop_length=self.hop_length)

    @staticmethod
    def istft(stft_matrix):
        """Compute Inverse Short-Time Fourier Transform (ISTFT)"""
        return librosa.istft(stft_matrix, hop_length=cfg.HOP_LENGTH)

    def compute_mdct(self):
        """Compute Modified Discrete Cosine Transform (MDCT) using STFT and DCT from librosa"""
        stft = self.stft(self.audio)
        magnitudes = np.abs(stft)
        phases = np.angle(stft)
        return magnitudes, phases

    def get_non_critical_coeffs(self, magnitudes, percentile=10):
        """Identify non-critical coefficients (lowest magnitude)"""
        threshold = np.percentile(np.abs(magnitudes.flatten()), percentile)
        mask = np.abs(magnitudes) < threshold
        return mask

    @staticmethod
    def reconstruct_audio(magnitudes, phases):
        """Reconstruct audio from magnitude/stft matrix and phase/non_critical_coeffs"""
        stft = magnitudes * np.exp(1j * phases)
        reconstructed_audio = AudioPreprocessor.istft(stft)
        return reconstructed_audio

    def plot_spectrogram(self, title):
        """Visualize audio spectrogram"""
        plt.figure(figsize=(10, 4))
        S = librosa.amplitude_to_db(np.abs(librosa.stft(self.audio)), ref=np.max)
        librosa.display.specshow(S, sr=self.sr, x_axis="time", y_axis="log")
        plt.colorbar(format="%+2.0f dB")
        plt.title(title)
        plt.tight_layout()
        plt.show()

    def save_audio(self, audio: np.ndarray, sr: int, path: str):
        wf.write(path, sr, (audio * NORM_NUM).astype(np.int16))
