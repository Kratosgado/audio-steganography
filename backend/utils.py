import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import soundfile as sf


def text_to_bits(text, length):
    bits = ''.join(f'{ord(c):08b}' for c in text)
    bits = bits.ljust(length, '0')[:length]
    return [int(b) for b in bits]


def bits_to_tensor(bits, audio_length):
    bit_array = np.array(bits).astype(np.float32)
    bit_array = np.repeat(bit_array[:, np.newaxis], audio_length // len(bits), axis=1)
    bit_array = bit_array.flatten()[:audio_length]
    return torch.tensor(bit_array).unsqueeze(0).unsqueeze(0)


def load_audio(path, target_sr=16000):
    waveform, sr = torchaudio.load(path)
    if sr != target_sr:
        waveform = T.Resample(sr, target_sr)(waveform)
    return waveform[:, :target_sr * 5]


def save_audio(tensor, path, sample_rate=16000):
    audio = tensor.squeeze().detach().cpu().numpy()
    sf.write(path, audio, sample_rate)