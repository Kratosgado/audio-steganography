import numpy as np
from core_modules.embedding_module import EmbeddingModule
from core_modules.preprocessor import AudioPreprocessor
from core_modules.config import cfg


class SignEncoding(EmbeddingModule):
    """Embeds and extracts messages from MDCT coefficients."""

    def __init__(self, steganalysis_net=None):
        pass

    def set_parameters(self, action):
        self.alpha = action[0]
        return self.alpha

    def embed(self, magnitudes, phases, mask, msg_bits: np.ndarray, **kwargs):
        """Embed message using sign encoding in non-critical coefficients"""
        # Apply mask to get non-critical coefficients
        coeffs = magnitudes.copy()
        non_critical = coeffs[mask]

        # Ensure we have enough coefficients for the message
        if len(non_critical) < len(msg_bits):
            raise ValueError("Message too long for available non-critical coefficients")

        # Embed message using sign encoding
        for i, bit in enumerate(msg_bits):
            sign = 1 if bit == 1 else -1
            non_critical[i] = sign * np.abs(non_critical[i]) * (1 + self.alpha)

        # Update coefficients
        coeffs[mask] = non_critical
        self.magnitudes = coeffs

        return AudioPreprocessor.reconstruct_audio(coeffs, phases)

    def extract(self, magnitudes, mask, message_length, **kwargs):
        """Extract message from non-critical coefficients"""
        # Apply mask to get non-critical coefficients
        non_critical = self.magnitudes[mask]

        # Extract message from sign
        msg_bits = []
        for i in range(message_length * 8):
            sign = 1 if non_critical[i] >= 0 else -1
            bit = 1 if sign > 0 else 0
            msg_bits.append(bit)

        # Convert binary to string
        return msg_bits
        # chars = [chr(int(bin_str[i:i+8], 2)) for i in range(0, len(bin_str), 8)]
