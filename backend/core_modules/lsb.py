from typing import List, Union
from core_modules.embedding_module import EmbeddingModule
import numpy as np


class AudioSteganography(EmbeddingModule):
    def __init__(self, embedding_method="lsb"):
        """
        Initialize steganography module with selected embedding method.

        Args:
            embedding_method (str): Method of embedding ('lsb' , 'spread_spectrum')
        """
        self.embedding_method = embedding_method

    def text_to_binary(self, message: str) -> List[int]:
        """
        Convert text message to binary representation
        Args:
            message (str): Input message to encode
        Returns:
            binary_message (List[int]): Binary representation of input message
        """
        return [int(bit) for char in message for bit in bin(ord(char))[2:].zfill(8)]

    def embed(self, audio_data: np.ndarray, msg_bits: np.ndarray) -> np.ndarray:
        """
        Embed message in audio data using LSB method
        Args:
            audio_data (np.ndarray): Audio data to embed message in
            msg_bits (np.ndarray): Message to embed in audio data
        Returns:
            stego_audio (np.ndarray): Audio data with embedded message
        """
        # Check if message can be embedded in audio data
        if len(msg_bits) > len(audio_data):
            raise ValueError("Message is too long to be embedded in audio data")

        stego_audio = audio_data.copy()

        # Embed message in audio data using LSB method
        for i, bit in enumerate(msg_bits):
            stego_audio[i] = (stego_audio[i] & ~1) | bit
        return stego_audio

    def extract(self, audio_data: np.ndarray, msg_length: int) -> str:
        """
        Extract message from audio data using LSB method
        Args:
            audio_data (np.ndarray): Audio data to extract message from
            msg_length (int): Length of message to extract
        Returns:
            extracted_message (str): Extracted message from audio data
        """
        extracted_bit = [audio_data[i] & 1 for i in range(msg_length)]

        # Convert binary representation to text message
        binary_str = "".join(map(str, extracted_bit))
        message = "".join(
            chr(int(binary_str[i : i + 8], 2)) for i in range(0, len(binary_str), 8)
        )
        return message

    def load_audio(self, file_path: str) -> np.ndarray:
        """
        Load audio file and convert to numpy array

        Args:
            file_path (str): Path to audio file

        Returns:
            np.ndarray: Audio data
        """
        audio_data, sample_rate = librosa.load(file_path)
        return audio_data

    def save_audio(
        self, audio_data: np.ndarray, file_path: str, sample_rate: int = 22050
    ):
        """
        Save audio data to file

        Args:
            audio_data (np.ndarray): Audio numpy array
            file_path (str): Output file path
            sample_rate (int): Audio sample rate
        """

        sf.write(file_path, audio_data, sample_rate)
