from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import soundfile as sf
import io
import os
import torch

from rl_environment import AudioStegEnvironment  # Ensure this exists and is functional
from rl_agent import QLearningAgent
from encoder import AudioStegEncoder
from decoder import AudioStegDecoder
from utils import text_to_bits, bits_to_tensor, load_audio, save_audio

app = FastAPI()

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========== Utility Functions ==========

def binary_string(message: str) -> np.ndarray:
    """Converts a string to a binary numpy array."""
    binary = ''.join(format(ord(c), '08b') for c in message)
    return np.array([int(bit) for bit in binary])

def spread_spectrum_embed(audio_data, message, seed=42):
    np.random.seed(seed)
    message_binary = ''.join(format(ord(char), '08b') for char in message)
    length_binary = format(len(message_binary), '016b')  # 16-bit length prefix
    full_binary = np.array([int(bit) for bit in length_binary + message_binary])
    
    pseudo_random_sequence = np.random.choice([-1, 1], size=len(audio_data))
    repeated_bits = np.tile(full_binary, len(audio_data) // len(full_binary) + 1)[:len(audio_data)]
    
    embedded_audio = audio_data + 0.01 * pseudo_random_sequence * repeated_bits
    return embedded_audio

def spread_spectrum_decode(audio_data, seed=42):
    np.random.seed(seed)
    pseudo_random_sequence = np.random.choice([-1, 1], size=len(audio_data))
    correlation = audio_data * pseudo_random_sequence
    
    chunk_size = len(audio_data) // 800  # better chunk estimate
    bits = [(1 if np.mean(correlation[i:i+chunk_size]) > 0 else 0)
            for i in range(0, len(audio_data), chunk_size)]
    
    # Decode length
    length_bits = bits[:16]
    msg_len = int(''.join(map(str, length_bits)), 2)
    data_bits = bits[16:16 + msg_len]

    chars = [chr(int(''.join(map(str, data_bits[i:i+8])), 2)) 
             for i in range(0, len(data_bits), 8) if len(data_bits[i:i+8]) == 8]
    return ''.join(chars)


# ========== Constants =======
ENCODING_METHODS = ["LSB", "Echo", "Phase", "SpreadSpectrum"]
AGENT_PATH = "agent_q_table.pkl"

# Load or initialize the agent
agent = QLearningAgent(actions=ENCODING_METHODS)
if os.path.exists(AGENT_PATH):
    agent.load(AGENT_PATH)

env = AudioStegEnvironment(ENCODING_METHODS)


# ========== Routes ==========

# Initialize models
encoder = AudioStegEncoder()
decoder = AudioStegDecoder()

@app.post("/upload")
async def embed_message(file: UploadFile = File(...), message: str = "Hello"):
    # Read and preprocess the audio file
    audio_data, samplerate = sf.read(io.BytesIO(await file.read()))
    
    # Ensure audio_data is float32 and properly shaped
    if len(audio_data.shape) == 1:
        audio_data = audio_data.astype(np.float32)
    else:
        audio_data = audio_data[:, 0].astype(np.float32)  # Take first channel if stereo
    
    # Convert to tensor with correct shape [1, 1, T]
    audio_tensor = torch.tensor(audio_data).unsqueeze(0).unsqueeze(0)
    
    # Convert message to binary and then to tensor
    message_bits = text_to_bits(message, len(audio_data))
    message_tensor = bits_to_tensor(message_bits, len(audio_data))
    
    # Pass audio and message tensors separately to match the encoder's forward method
    stego_audio = encoder(audio_tensor, message_tensor)
    
    # Save the processed audio
    output_file = "output.flac"
    save_audio(stego_audio, output_file, samplerate)
    
    return FileResponse(output_file, media_type="audio/flac", filename=output_file)

@app.post("/decode")
async def decode_message(file: UploadFile = File(...)):
    # Read and preprocess the audio file
    audio_data, samplerate = sf.read(io.BytesIO(await file.read()))
    
    # Ensure audio_data is float32 and properly shaped
    if len(audio_data.shape) == 1:
        audio_data = audio_data.astype(np.float32)
    else:
        audio_data = audio_data[:, 0].astype(np.float32)  # Take first channel if stereo
    
    # Convert to tensor with correct shape [1, 1, T]
    stego_audio = torch.tensor(audio_data).unsqueeze(0).unsqueeze(0)
    
    # Print shape for debugging
    print(f"Stego audio tensor shape: {stego_audio.shape}")
    
    # Decode the message
    decoded_bits = decoder(stego_audio)
    
    # Convert bits to text (you'll need to implement this conversion)
    # For now, return the raw bits as a list
    # print(decoded_bits.squeeze().tolist())
    # return {"decoded_bits": decoded_bits.squeeze().tolist()}
    
    # Convert bits to text
    decoded_message = ""
    for i in range(0, decoded_bits.size(-1), 8):
        if i + 8 <= decoded_bits.size(-1):
            byte_bits = decoded_bits[0, 0, i:i+8]
            byte = int(''.join(map(lambda x: '1' if x > 0.5 else '0', byte_bits.tolist())), 2)
            if byte < 128:  # Only accept ASCII characters
                decoded_message += chr(byte)
            else:
                break  # Stop at first non-ASCII character
    print(decoded_message)
    return JSONResponse(content={"decoded_message": decoded_message})


@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    try:
        audio_data, _ = sf.read(io.BytesIO(await file.read()))
        tensor_audio = torch.tensor(audio_data).unsqueeze(0)

        env = AudioStegEnvironment()
        features = env.extract_audio_features(tensor_audio)

        n_fft = 1024
        hop = 512
        spec = torch.stft(tensor_audio.squeeze(0), n_fft, hop, return_complex=True)
        magnitude = torch.abs(spec)

        low = torch.mean(magnitude[:n_fft//6]).item()
        mid = torch.mean(magnitude[n_fft//6:n_fft//3]).item()
        high = torch.mean(magnitude[n_fft//3:]).item()
        total = low + mid + high

        low_ratio = low / total
        mid_ratio = mid / total
        high_ratio = high / total

        eps = 1e-10
        log_mag = torch.log(magnitude + eps)
        flatness = torch.exp(torch.mean(log_mag)) / (torch.mean(magnitude) + eps)
        flat_val = flatness.item()

        # Heuristic detection
        expected = {"low": 0.5, "mid": 0.3, "high": 0.2}
        deviation = abs(low_ratio - expected["low"]) + \
                    abs(mid_ratio - expected["mid"]) + \
                    abs(high_ratio - expected["high"])
        steg_score = min(1.0, (0.3 if flat_val < 0.5 else 0) + min(0.7, deviation * 2))

        confidence = "Low" if steg_score < 0.4 else "Medium" if steg_score < 0.7 else "High"

        return JSONResponse(content={
            "analysis_results": {
                "spectral_features": features.tolist(),
                "spectral_flatness": flat_val,
                "energy_distribution": {
                    "low_band": low_ratio,
                    "mid_band": mid_ratio,
                    "high_band": high_ratio,
                },
                "steg_likelihood": steg_score,
                "detection_confidence": confidence
            }
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# ========== Training Routes =========_
@app.post("/train_agent")
def train_agent(episodes: int = 10):
    for ep in range(episodes):
        state = env.reset()
        for _ in range(10):  # steps per episode
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state
            if done:
                break
    agent.save(AGENT_PATH)
    return {"status": "training complete", "episodes": episodes}

@app.get("/best_method")
def get_best_method():
    """Returns the best encoding method based on learned Q-values."""
    scores = {a: agent.get_q("LSB", a) for a in ENCODING_METHODS}
    best_method = max(scores, key=scores.get)
    return {"best_method": best_method, "scores": scores}

# ========== Entry Point ==========

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
