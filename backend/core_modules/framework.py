import numpy as np
from core_modules.preprocessor import AudioPreprocessor
from core_modules.embedding_module import EmbeddingModule
from core_modules.sign_encoding import SignEncoding
from core_modules.spread_spectrum import SpreadSpectrum
from core_modules.message_processor import string_to_bits, bits_to_string
from core_modules.audio_analyzer import AudioAnalyzer

from core_modules.config import Config, cfg


class RLAudioSteganography:
    """Main framework class"""

    def __init__(self) -> None:
        self.cfg = cfg
        self.audio_analyzer = AudioAnalyzer()

    def Initialize_components(self, method="sign-encoding"):
        """Initialize components"""
        # self.original_magnitudes, self.phases = self.preprocessor.compute_mdct()
        # self.mask = self.preprocessor.get_non_critical_coeffs(self.original_magnitudes)
        self.method = method
        self.embedder: EmbeddingModule = (
            SignEncoding() if method == "sign-encoding" else SpreadSpectrum()
        )

    def embed_message(self, waveform, sr, message, model):
        """Embed a message into an audio file using trained policy"""

        waveform = AudioPreprocessor.resample_audio(waveform, sr)

        msg_bits = string_to_bits(message)
        obs = AudioPreprocessor.extract_audio_features(waveform, len(msg_bits))
        action, _ = model.predict(obs, deterministic=True)
        action = self.embedder.set_parameters(action[0])
        # Convert message string to bits for embedding
        stego_waveform = self.embedder.embed(
            msg_bits=msg_bits, action=action, waveform=waveform
        )

        # if self.method == "spread-spectrum":
        #     # Calculate optimal action values based on audio features
        #     print(f"Analyzing audio features for optimal parameters...")
        #     action = self.audio_analyzer.calculate_optimal_actions(
        #         audio_data, cfg.SAMPLE_RATE, message_length=len(message)
        #     )
        #     print(f"Calculated optimal actions: {action}")
        # else:
        #     action = [0.1]

        return stego_waveform

    def extract_message(self, stego_data, sr, msg_length=None):
        """Extract a message from a stego audio file"""
        waveform = AudioPreprocessor.resample_audio(stego_data, sr)
        extracted_bits = self.embedder.extract(
            stego_waveform=waveform, message_length=msg_length
        )
        return bits_to_string(extracted_bits)

    def get_audio_analysis(self, audio_data):
        """Get comprehensive audio analysis and capacity estimates"""
        return self.audio_analyzer.get_embedding_capacity_estimate(
            audio_data, cfg.SAMPLE_RATE
        )
