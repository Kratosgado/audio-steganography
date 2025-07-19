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
        self.embedded_mask = None  # Store the mask used during embedding
        self.original_magnitudes = None  # Store original magnitudes
        self.audio_analyzer = AudioAnalyzer()

    def Initialize_components(self, audio_data, method="sign-encoding"):
        """Initialize components"""
        self.preprocessor = AudioPreprocessor(audio_data=audio_data)
        # self.original_magnitudes, self.phases = self.preprocessor.compute_mdct()
        # self.mask = self.preprocessor.get_non_critical_coeffs(self.original_magnitudes)
        self.method = method
        self.embedder: EmbeddingModule = (
            SignEncoding() if method == "sign-encoding" else SpreadSpectrum()
        )

    def embed_message(self, audio_data, message):
        """Embed a message into an audio file using trained policy"""
        # Re-initialize preprocessor and compute magnitudes/phases/mask for the specific audio being embedded into
        self.preprocessor = AudioPreprocessor(audio_data=audio_data)
        # self.original_magnitudes, self.phases = self.preprocessor.compute_mdct()
        # self.mask = self.preprocessor.get_non_critical_coeffs(self.original_magnitudes)
        # self.embedded_mask = self.mask  # Store the mask used for embedding

        # Steganalysis network is initialized in Initialize_components, ensure it's done before embedding
        # If embed_message is called standalone, ensure Initialize_components is called first with the correct audio_path

        msg_bits = string_to_bits(message)
        
        if self.method == "spread-spectrum":
            # Calculate optimal action values based on audio features
            print(f"Analyzing audio features for optimal parameters...")
            action = self.audio_analyzer.calculate_optimal_actions(
                audio_data, 
                cfg.SAMPLE_RATE, 
                message_length=len(message)
            )
            print(f"Calculated optimal actions: {action}")
        else:
            action = [0.1]
            
        self.embedder.set_parameters(action)
        
        # Convert message string to bits for embedding
        message_bits = string_to_bits(message)
        
        # stego_audio = self.embedder.embed(self.original_magnitudes, self.embedded_mask, message_bits)
        stego_audio = self.embedder.embed(
            magnitudes=self.original_magnitudes,
            # phases=self.phases,
            # mask=self.mask,
            msg_bits=msg_bits,
            original_audio=self.preprocessor.audio.copy(),
            action=action,
        )
        return stego_audio

    def extract_message(self, stego_data, msg_length=None):
        """Extract a message from a stego audio file"""
        # Load the stego audio
        preprocessor_stego = AudioPreprocessor(audio_data=stego_data)
        # magnitudes_stego, phases_stego = preprocessor_stego.compute_mdct()

        # Use the stored mask from embedding for extraction
        # if self.embedded_mask is None:
        #     raise ValueError(
        #         "Embedding must be performed before extraction to get the mask."
        #     )

        # extracted_message = self.embedder.extract(magnitudes_stego, self.embedded_mask, msg_length)
        extracted_bits = self.embedder.extract(
            stego_audio=preprocessor_stego.audio.copy(),
            # magnitudes=magnitudes_stego,
            # mask=self.mask,
            message_length=msg_length,
        )
        return bits_to_string(extracted_bits)
    
    def get_audio_analysis(self, audio_data):
        """Get comprehensive audio analysis and capacity estimates"""
        return self.audio_analyzer.get_embedding_capacity_estimate(audio_data, cfg.SAMPLE_RATE)
