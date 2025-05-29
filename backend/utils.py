import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import soundfile as sf
import os

def text_to_bits(text, length=None):
    """Convert text to binary bits with optional length specification"""
    if not text:  # Add this check for empty text
        text = "Empty"  # Use a default message instead of empty string
        
    if isinstance(text, str):
        bits = ''.join(f'{ord(c):08b}' for c in text)
    else:
        # If already bits, return as is
        return text if isinstance(text, list) else [int(b) for b in str(text)]
    
    bit_list = [int(b) for b in bits]
    
    if length is not None:
        if len(bit_list) < length:
            # Pad with zeros
            bit_list.extend([0] * (length - len(bit_list)))
        else:
            # Truncate
            bit_list = bit_list[:length]
    
    return bit_list

def bits_to_tensor(bits, audio_length):
    """Convert bits to tensor matching audio length"""
    if not bits:
        bits = [0, 1] * (audio_length // 2)  # Default pattern
    
    bit_array = np.array(bits).astype(np.float32)
    
    # Repeat or truncate to match audio length
    if len(bit_array) < audio_length:
        repeat_factor = audio_length // len(bit_array) + 1
        bit_array = np.tile(bit_array, repeat_factor)
    
    bit_array = bit_array[:audio_length]
    return torch.tensor(bit_array).unsqueeze(0)

def load_audio(path, target_sr=16000, max_duration=5):
    """Load audio file with error handling"""
    try:
        if not os.path.exists(path):
            print(f"Warning: Audio file {path} not found. Using synthetic audio.")
            return torch.randn(1, target_sr * max_duration)
        
        waveform, sr = torchaudio.load(path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if necessary
        if sr != target_sr:
            resampler = T.Resample(sr, target_sr)
            waveform = resampler(waveform)
        
        # Limit duration
        max_samples = target_sr * max_duration
        if waveform.shape[1] > max_samples:
            waveform = waveform[:, :max_samples]
        elif waveform.shape[1] < max_samples:
            # Pad with zeros if too short
            padding = max_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        
        return waveform
        
    except Exception as e:
        print(f"Error loading audio {path}: {e}. Using synthetic audio.")
        return torch.randn(1, target_sr * max_duration)

def save_audio(tensor, path, sample_rate=16000):
    """Save audio tensor to file"""
    try:
        # Ensure tensor is on CPU and detached
        if isinstance(tensor, torch.Tensor):
            audio = tensor.squeeze().detach().cpu().numpy()
        else:
            audio = np.array(tensor).squeeze()
        
        # Normalize to prevent clipping
        if np.max(np.abs(audio)) > 1.0:
            audio = audio / np.max(np.abs(audio))
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        sf.write(path, audio, sample_rate)
        return True
    except Exception as e:
        print(f"Error saving audio to {path}: {e}")
        return False

def simple_lsb_embed(audio_data, message, bits_per_sample=1):
    """Simple LSB steganography that actually works"""
    try:
        # Ensure audio_data is numpy array
        if isinstance(audio_data, torch.Tensor):
            audio_data = audio_data.detach().cpu().numpy()
        
        audio_data = np.array(audio_data).flatten()
        
        # Convert message to binary
        if isinstance(message, str):
            message_binary = ''.join(format(ord(c), '08b') for c in message)
        else:
            message_binary = ''.join(str(b) for b in message)
        
        # Add end marker
        message_binary += '1111111111111110'  # 16-bit end marker
        
        # Convert audio to 16-bit integers for LSB manipulation
        audio_int = (audio_data * 32767).astype(np.int16)
        
        if len(message_binary) > len(audio_int) * bits_per_sample:
            print("Warning: Message too long for audio file, truncating")
            message_binary = message_binary[:len(audio_int) * bits_per_sample]
        
        # Embed message in LSBs
        for i, bit in enumerate(message_binary):
            if i >= len(audio_int):
                break
            # Clear LSB and set to message bit
            audio_int[i] = (audio_int[i] & ~1) | int(bit)
        
        # Convert back to float
        return audio_int.astype(np.float32) / 32767.0
        
    except Exception as e:
        print(f"LSB embedding error: {e}")
        return audio_data

def simple_lsb_extract(audio_data):
    """Extract message from LSB steganography"""
    try:
        # Ensure audio_data is numpy array
        if isinstance(audio_data, torch.Tensor):
            audio_data = audio_data.detach().cpu().numpy()
        
        audio_data = np.array(audio_data).flatten()
        
        # Convert to 16-bit integers
        audio_int = (audio_data * 32767).astype(np.int16)
        
        # Extract LSBs
        bits = []
        for sample in audio_int:
            bits.append(str(sample & 1))
        
        # Convert bits to string
        binary_string = ''.join(bits)
        
        # Look for end marker
        end_marker = '1111111111111110'
        end_pos = binary_string.find(end_marker)
        
        if end_pos == -1:
            # If no end marker found, try to decode what we have
            end_pos = len(binary_string) - (len(binary_string) % 8)
            end_pos = min(end_pos, 1000)  # Limit length
        
        # Extract message bits (excluding end marker)
        message_bits = binary_string[:end_pos]
        
        # Convert to characters
        message = ""
        for i in range(0, len(message_bits), 8):
            if i + 8 <= len(message_bits):
                byte = message_bits[i:i+8]
                if len(byte) == 8:
                    try:
                        char_code = int(byte, 2)
                        if 32 <= char_code <= 126:  # Printable ASCII
                            message += chr(char_code)
                        else:
                            break  # Stop at non-printable character
                    except ValueError:
                        break
        
        return message
        
    except Exception as e:
        print(f"LSB extraction error: {e}")
        return ""

def create_synthetic_audio(duration=5, sample_rate=16000, frequency=440):
    """Create synthetic audio for testing when real audio files are not available"""
    t = torch.linspace(0, duration, sample_rate * duration)
    # Create a mix of frequencies to make it more realistic
    audio = (torch.sin(2 * np.pi * frequency * t) + 
             0.5 * torch.sin(2 * np.pi * frequency * 2 * t) + 
             0.25 * torch.sin(2 * np.pi * frequency * 3 * t))
    
    # Add some noise
    audio += 0.1 * torch.randn_like(audio)
    
    # Normalize
    audio = audio / torch.max(torch.abs(audio))
    
    return audio.unsqueeze(0)


def bits_to_tensor(bits, audio_length):
    """Convert bits to tensor matching audio length"""
    if not bits:
        bits = [0, 1] * (audio_length // 2)  # Default pattern
    
    bit_array = np.array(bits).astype(np.float32)
    
    # Repeat or truncate to match audio length
    if len(bit_array) < audio_length:
        repeat_factor = audio_length // len(bit_array) + 1
        bit_array = np.tile(bit_array, repeat_factor)
    
    bit_array = bit_array[:audio_length]
    # Add an extra dimension to match audio tensor shape [1, 1, T]
    return torch.tensor(bit_array).unsqueeze(0).unsqueeze(0)


def text_to_bits(text, length=None):
    """Convert text to binary bits with optional length specification"""
    if not text:  # Add this check for empty text
        text = "Empty"  # Use a default message instead of empty string
        
    if isinstance(text, str):
        bits = ''.join(f'{ord(c):08b}' for c in text)
    else:
        # If already bits, return as is
        return text if isinstance(text, list) else [int(b) for b in str(text)]
    
    bit_list = [int(b) for b in bits]
    
    if length is not None:
        if len(bit_list) < length:
            # Pad with zeros
            bit_list.extend([0] * (length - len(bit_list)))
        else:
            # Truncate
            bit_list = bit_list[:length]
    
    return bit_list

def bits_to_tensor(bits, audio_length):
    """Convert bits to tensor matching audio length - FIXED VERSION"""
    if not bits or len(bits) == 0:
        # Create a default pattern if bits is empty
        bits = [0, 1] * (audio_length // 2)
        if len(bits) < audio_length:
            bits.extend([0] * (audio_length - len(bits)))
    
    # Ensure bits is a list of integers
    if isinstance(bits, str):
        bits = [int(b) for b in bits]
    elif not isinstance(bits, list):
        bits = list(bits)
    
    # Convert to numpy array
    bit_array = np.array(bits, dtype=np.float32)
    
    # Handle case where bit_array might be empty
    if bit_array.size == 0:
        bit_array = np.zeros(audio_length, dtype=np.float32)
    
    # Repeat or truncate to match audio length
    if len(bit_array) < audio_length:
        if len(bit_array) > 0:
            repeat_factor = audio_length // len(bit_array) + 1
            bit_array = np.tile(bit_array, repeat_factor)
        else:
            bit_array = np.zeros(audio_length, dtype=np.float32)
    
    # Ensure we have exactly audio_length samples
    bit_array = bit_array[:audio_length]
    
    # Convert to tensor with correct shape [1, 1, T]
    return torch.tensor(bit_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

def load_audio(path, target_sr=16000, max_duration=5):
    """Load audio file with error handling"""
    try:
        if not os.path.exists(path):
            print(f"Warning: Audio file {path} not found. Using synthetic audio.")
            return torch.randn(1, target_sr * max_duration)
        
        waveform, sr = torchaudio.load(path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if necessary
        if sr != target_sr:
            resampler = T.Resample(sr, target_sr)
            waveform = resampler(waveform)
        
        # Limit duration
        max_samples = target_sr * max_duration
        if waveform.shape[1] > max_samples:
            waveform = waveform[:, :max_samples]
        elif waveform.shape[1] < max_samples:
            # Pad with zeros if too short
            padding = max_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        
        return waveform
        
    except Exception as e:
        print(f"Error loading audio {path}: {e}. Using synthetic audio.")
        return torch.randn(1, target_sr * max_duration)

def save_audio(tensor, path, sample_rate=16000):
    """Save audio tensor to file"""
    try:
        # Ensure tensor is on CPU and detached
        if isinstance(tensor, torch.Tensor):
            audio = tensor.squeeze().detach().cpu().numpy()
        else:
            audio = np.array(tensor).squeeze()
        
        # Normalize to prevent clipping
        if np.max(np.abs(audio)) > 1.0:
            audio = audio / np.max(np.abs(audio))
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        sf.write(path, audio, sample_rate)
        return True
    except Exception as e:
        print(f"Error saving audio to {path}: {e}")
        return False

def simple_lsb_embed(audio_data, message, bits_per_sample=1):
    """Simple LSB steganography that actually works"""
    try:
        # Ensure audio_data is numpy array
        if isinstance(audio_data, torch.Tensor):
            audio_data = audio_data.detach().cpu().numpy()
        
        audio_data = np.array(audio_data).flatten()
        
        # Convert message to binary
        if isinstance(message, str):
            message_binary = ''.join(format(ord(c), '08b') for c in message)
        else:
            message_binary = ''.join(str(b) for b in message)
        
        # Add end marker
        message_binary += '1111111111111110'  # 16-bit end marker
        
        # Convert audio to 16-bit integers for LSB manipulation
        audio_int = (audio_data * 32767).astype(np.int16)
        
        if len(message_binary) > len(audio_int) * bits_per_sample:
            print("Warning: Message too long for audio file, truncating")
            message_binary = message_binary[:len(audio_int) * bits_per_sample]
        
        # Embed message in LSBs
        for i, bit in enumerate(message_binary):
            if i >= len(audio_int):
                break
            # Clear LSB and set to message bit
            audio_int[i] = (audio_int[i] & ~1) | int(bit)
        
        # Convert back to float
        return audio_int.astype(np.float32) / 32767.0
        
    except Exception as e:
        print(f"LSB embedding error: {e}")
        return audio_data

def simple_lsb_extract(audio_data):
    """Extract message from LSB steganography"""
    try:
        # Ensure audio_data is numpy array
        if isinstance(audio_data, torch.Tensor):
            audio_data = audio_data.detach().cpu().numpy()
        
        audio_data = np.array(audio_data).flatten()
        
        # Convert to 16-bit integers
        audio_int = (audio_data * 32767).astype(np.int16)
        
        # Extract LSBs
        bits = []
        for sample in audio_int:
            bits.append(str(sample & 1))
        
        # Convert bits to string
        binary_string = ''.join(bits)
        
        # Look for end marker
        end_marker = '1111111111111110'
        end_pos = binary_string.find(end_marker)
        
        if end_pos == -1:
            # If no end marker found, try to decode what we have
            end_pos = len(binary_string) - (len(binary_string) % 8)
            end_pos = min(end_pos, 1000)  # Limit length
        
        # Extract message bits (excluding end marker)
        message_bits = binary_string[:end_pos]
        
        # Convert to characters
        message = ""
        for i in range(0, len(message_bits), 8):
            if i + 8 <= len(message_bits):
                byte = message_bits[i:i+8]
                if len(byte) == 8:
                    try:
                        char_code = int(byte, 2)
                        if 32 <= char_code <= 126:  # Printable ASCII
                            message += chr(char_code)
                        else:
                            break  # Stop at non-printable character
                    except ValueError:
                        break
        
        return message
        
    except Exception as e:
        print(f"LSB extraction error: {e}")
        return ""

def create_synthetic_audio(duration=5, sample_rate=16000, frequency=440):
    """Create synthetic audio for testing when real audio files are not available"""
    t = torch.linspace(0, duration, sample_rate * duration)
    # Create a mix of frequencies to make it more realistic
    audio = (torch.sin(2 * np.pi * frequency * t) + 
             0.5 * torch.sin(2 * np.pi * frequency * 2 * t) + 
             0.25 * torch.sin(2 * np.pi * frequency * 3 * t))
    
    # Add some noise
    audio += 0.1 * torch.randn_like(audio)
    
    # Normalize
    audio = audio / torch.max(torch.abs(audio))
    
    return audio.unsqueeze(0)