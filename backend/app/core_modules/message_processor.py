import numpy as np


def string_to_bits(message: str) -> np.ndarray:
    """Convert a string to a sequence of bits."""
    return np.array(
        [int(bit) for bit in "".join(format(ord(char), "08b") for char in message)],
        dtype=np.float32,
    )


def bits_to_string(bits) -> str:
    """Convert a sequence of bits to a string."""
    text = ""
    for bit in range(0, len(bits), 8):
        if bit + 8 <= len(bits):
            byte = "".join(str(bit) for bit in bits[bit : bit + 8])
            text += chr(int(byte, 2))
    return text


def validate_message_length(message: str, audio_length: int, sample_rate: int) -> bool:
    """
    Validate if a message can be embedded in the given audio
    
    Args:
        message: Message to embed
        audio_length: Length of audio in samples
        sample_rate: Sample rate of audio
        
    Returns:
        True if message can be embedded, False otherwise
    """
    # Conservative estimate: 1 bit per 100 samples for spread spectrum
    max_bits = audio_length // 100
    message_bits = len(message) * 8
    
    return message_bits <= max_bits


def calculate_embedding_efficiency(message: str, audio_length: int, sample_rate: int) -> float:
    """
    Calculate embedding efficiency (bits per sample)
    
    Args:
        message: Message to embed
        audio_length: Length of audio in samples
        sample_rate: Sample rate of audio
        
    Returns:
        Embedding efficiency as bits per sample
    """
    message_bits = len(message) * 8
    return message_bits / audio_length