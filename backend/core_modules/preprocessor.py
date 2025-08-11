import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from core_modules.config import cfg

NORM_NUM = 32768


class AudioPreprocessor:
    """Handles audio loading, MDCT, and inverse MDCT with sign preservation."""

    @staticmethod
    def load_audio(path):
        """Load WAV audio file and resample to cfg.SAMPLE_RATE"""
        audio, _ = librosa.load(path, sr=cfg.SAMPLE_RATE)
        return audio

    @staticmethod
    def resample_audio(waveform, sr):
        """Resample audio to cfg.SAMPLE_RATE"""
        audio = librosa.resample(waveform, orig_sr=sr, target_sr=cfg.SAMPLE_RATE)
        return audio

    @staticmethod
    def normalize_audio(waveform):
        """Normalize audio to [-1, 1]"""
        return waveform / np.max(np.abs(waveform))

    @staticmethod
    def stft(waveform: np.ndarray):
        """Compute Short-Time Fourier Transform (STFT)"""
        return librosa.stft(waveform, n_fft=cfg.FRAME_SIZE, hop_length=cfg.HOP_LENGTH)

    @staticmethod
    def istft(stft_matrix, length):
        """Compute Inverse Short-Time Fourier Transform (ISTFT)"""
        return librosa.istft(
            stft_matrix, hop_length=cfg.HOP_LENGTH, n_fft=cfg.FRAME_SIZE, length=length
        )

    @staticmethod
    def compute_mdct(waveform):
        """
        Compute Modified Discrete Cosine Transform (MDCT) using STFT and DCT.
        Returns both magnitudes and phases to preserve sign information.
        """
        stft = AudioPreprocessor.stft(waveform)
        magnitudes = np.abs(stft)
        phases = np.angle(stft)
        return magnitudes, phases

    @staticmethod
    def get_non_critical_coeffs(mdct_coeffs, percentile=10):
        """Identify non-critical coefficients (lowest magnitude)"""
        magnitudes = np.abs(mdct_coeffs)
        threshold = np.percentile(magnitudes.flatten(), percentile)
        return magnitudes < threshold

    @staticmethod
    def reconstruct_audio(magnitudes, phases, length):
        """
        Reconstruct audio from magnitude and phase.
        The phase information preserves the signs even if magnitudes are modified.
        """
        # Reconstruct complex STFT
        stft = magnitudes * np.exp(1j * phases)

        # Inverse STFT to get audio
        reconstructed_audio = AudioPreprocessor.istft(stft, length)
        return reconstructed_audio

    @staticmethod
    def extract_audio_features(waveform, bits_len):
        features = {}
        features["mfcc"] = librosa.feature.mfcc(
            y=waveform,
            sr=cfg.SAMPLE_RATE,
            n_mfcc=cfg.N_MELS,
            n_fft=cfg.FRAME_SIZE,
            hop_length=cfg.HOP_LENGTH,
        ).mean(axis=1)

        features["spectral_centroid"] = librosa.feature.spectral_centroid(
            y=waveform, sr=cfg.SAMPLE_RATE
        ).mean(keepdims=True)

        features["rms"] = librosa.feature.rms(y=waveform).mean(keepdims=True)
        features["zcr"] = librosa.feature.zero_crossing_rate(y=waveform).mean(
            keepdims=True
        )
        features["bits_len"] = bits_len / 1000

        return features

    @staticmethod
    def save_audio(audio: np.ndarray, sr: int, path: str):
        """Save audio to file"""
        sf.write(
            path, audio, sr
        )  # wf.write(path, sr, (audio * NORM_NUM).astype(np.int16))
