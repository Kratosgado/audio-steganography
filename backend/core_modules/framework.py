import numpy as np
from preprocessor import AudioPreprocessor
from sign_encoding import SignEncoding
from spread_spectrum import SpreadSpectrum
from backend.core_modules.config import Config, cfg
from embedding_module import EmbeddingModule

from fastapi import UploadFile
device = "cpu"

class RLAudioSteganography:
  """Main framework class"""
  def __init__(self, cfg: Config) -> None:
    self.cfg = cfg
    self.embedded_mask = None # Store the mask used during embedding
    self.original_magnitudes = None # Store original magnitudes

  def Initialize_components(self, audio_file: UploadFile, method="sign-encoding"):
    """Initialize components"""
    self.preprocessor = AudioPreprocessor(audio_path=audio_file)
    self.original_magnitudes, self.phases = self.preprocessor.compute_mdct()
    self.mask = self.preprocessor.get_non_critical_coeffs(self.original_magnitudes)
    self.method = method
    self.embedder: EmbeddingModule = SignEncoding() if method == "sign-encoding" else SpreadSpectrum()

  @staticmethod
  def string_to_bits(message):
    """Convert a string to a sequence of bits."""
    return np.array([int(bit) for bit in ''.join(format(ord(char), '08b') for char in message)], dtype=np.float32)

  @staticmethod
  def bits_to_string(bits):
    """Convert a sequence of bits to a string."""
    text = ''
    for bit in range(0, len(bits), 8):
      if bit + 8 <= len(bits):
        byte = ''.join(str(bit) for bit in bits[bit:bit + 8])
        text += chr(int(byte, 2))
    return text

  def embed_message(self, audio_data, message):
    """Embed a message into an audio file using trained policy"""
    # Re-initialize preprocessor and compute magnitudes/phases/mask for the specific audio being embedded into
    self.preprocessor = AudioPreprocessor(audio_data=audio_data)
    self.original_magnitudes, self.phases = self.preprocessor.compute_mdct()
    self.mask = self.preprocessor.get_non_critical_coeffs(self.original_magnitudes)
    self.embedded_mask = self.mask # Store the mask used for embedding

    # Steganalysis network is initialized in Initialize_components, ensure it's done before embedding
    # If embed_message is called standalone, ensure Initialize_components is called first with the correct audio_path

    msg_bits = self.string_to_bits(message)
    action = [10000, 50, 20]
    self.embedder.set_parameters(action)
    # Convert message string to bits for embedding
    message_bits = self.string_to_bits(message)
    # stego_audio = self.embedder.embed(self.original_magnitudes, self.embedded_mask, message_bits)
    stego_audio = self.embedder.embed(
        magnitudes = self.original_magnitudes,
        phases=self.phases,
        mask = self.mask,
        msg_bits = msg_bits,
        original_audio=self.preprocessor.audio.copy(),
        action=action)
    return stego_audio


  def extract_message(self, stego_data, msg_length):
    """Extract a message from a stego audio file"""
    # Load the stego audio
    preprocessor_stego = AudioPreprocessor(audio_data=stego_data)
    magnitudes_stego, phases_stego = preprocessor_stego.compute_mdct()

    # Use the stored mask from embedding for extraction
    if self.embedded_mask is None:
        raise ValueError("Embedding must be performed before extraction to get the mask.")

    # extracted_message = self.embedder.extract(magnitudes_stego, self.embedded_mask, msg_length)
    extracted_bits = self.embedder.extract(
        stego_audio =preprocessor_stego.audio.copy(),
        magnitudes=magnitudes_stego,
        mask=self.mask,
        message_length= msg_length)
    return self.bits_to_string(extracted_bits)

  def plot_training_history(self):
    """Plot training metrics"""
    # plt.figure(figsize=(12, 8))
    # plt.subplot(2, 2, 1)
    # plt.plot(self.training_history['rewards'], label='Reward')
    # plt.title('Training Rewards')
    # plt.xlabel('Episode')
    # plt.ylabel('Average Reward')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()