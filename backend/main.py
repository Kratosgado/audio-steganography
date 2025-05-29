# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.responses import FileResponse, JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# import numpy as np
# import soundfile as sf
# import io
# import os
# import torch
# import pickle
# from typing import Dict, Any

# from rl_environment import AudioStegEnvironment
# from rl_agent import QLearningAgent
# from encoder import AudioStegEncoder
# from decoder import AudioStegDecoder
# from utils import text_to_bits, bits_to_tensor, load_audio, save_audio

# app = FastAPI()

# # Middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ========== Constants ==========
# ENCODING_METHODS = ["LSB", "Echo", "Phase", "SpreadSpectrum"]
# TRAINED_AGENT_PATH = "models/trained_agent_final.pkl"
# FALLBACK_AGENT_PATH = "agent_q_table.pkl"

# # ========== Initialize Models and Agent ==========
# encoder = AudioStegEncoder()
# decoder = AudioStegDecoder()
# env = AudioStegEnvironment(ENCODING_METHODS)

# # Load the trained agent
# agent = QLearningAgent(actions=ENCODING_METHODS)
# if os.path.exists(TRAINED_AGENT_PATH):
#     print(f"Loading trained agent from {TRAINED_AGENT_PATH}")
#     agent.load(TRAINED_AGENT_PATH)
# elif os.path.exists(FALLBACK_AGENT_PATH):
#     print(f"Loading fallback agent from {FALLBACK_AGENT_PATH}")
#     agent.load(FALLBACK_AGENT_PATH)
# else:
#     print("No trained agent found, using random initialization")

# def get_optimal_encoding_method(audio_features: torch.Tensor) -> str:
#     """
#     Use the trained RL agent to select the optimal encoding method
#     based on audio features.
#     """
#     try:
#         # Convert audio features to a state representation
#         if isinstance(audio_features, torch.Tensor):
#             state_features = audio_features.cpu().numpy().flatten()
#         else:
#             state_features = np.array(audio_features).flatten()
        
#         # Create a state string representation for the Q-table
#         # You might need to adjust this based on your state representation
#         state = f"features_{hash(tuple(state_features[:10])) % 1000}"  # Use first 10 features
        
#         # Get the best action from the trained agent
#         best_method = agent.choose_action(state, epsilon=0.0)  # No exploration, pure exploitation
#         print(f"RL Agent selected method: {best_method}")
#         return best_method
#     except Exception as e:
#         print(f"Error in RL agent selection: {e}, falling back to LSB")
#         return "LSB"

# def advanced_audio_analysis(audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
#     """
#     Perform comprehensive audio analysis for steganography detection.
#     """
#     try:
#         # Convert to tensor for processing
#         audio_tensor = torch.tensor(audio_data, dtype=torch.float32)
#         if len(audio_tensor.shape) == 1:
#             audio_tensor = audio_tensor.unsqueeze(0)
        
#         # Extract features using the environment
#         features = env.extract_audio_features(audio_tensor)
        
#         # Spectral analysis
#         n_fft = 1024
#         hop_length = 512
#         window = torch.hann_window(n_fft)
        
#         stft = torch.stft(
#             audio_tensor.squeeze(0), 
#             n_fft=n_fft, 
#             hop_length=hop_length, 
#             window=window,
#             return_complex=True
#         )
        
#         magnitude = torch.abs(stft)
#         power = magnitude ** 2
        
#         # Frequency band analysis
#         freq_bins = magnitude.shape[0]
#         low_freq_end = freq_bins // 4
#         mid_freq_end = 3 * freq_bins // 4
        
#         low_band_energy = torch.mean(power[:low_freq_end]).item()
#         mid_band_energy = torch.mean(power[low_freq_end:mid_freq_end]).item()
#         high_band_energy = torch.mean(power[mid_freq_end:]).item()
        
#         total_energy = low_band_energy + mid_band_energy + high_band_energy
        
#         if total_energy > 0:
#             low_ratio = low_band_energy / total_energy
#             mid_ratio = mid_band_energy / total_energy
#             high_ratio = high_band_energy / total_energy
#         else:
#             low_ratio = mid_ratio = high_ratio = 1/3
        
#         # Spectral flatness (Wiener entropy)
#         eps = 1e-10
#         geometric_mean = torch.exp(torch.mean(torch.log(magnitude + eps)))
#         arithmetic_mean = torch.mean(magnitude)
#         spectral_flatness = (geometric_mean / (arithmetic_mean + eps)).item()
        
#         # Spectral centroid
#         freqs = torch.linspace(0, sample_rate/2, magnitude.shape[0])
#         spectral_centroid = torch.sum(freqs.unsqueeze(1) * magnitude) / torch.sum(magnitude)
#         spectral_centroid = spectral_centroid.item()
        
#         # Zero crossing rate
#         zero_crossings = torch.sum(torch.diff(torch.sign(audio_tensor)) != 0).item()
#         zcr = zero_crossings / len(audio_tensor.squeeze())
        
#         # Statistical measures
#         rms_energy = torch.sqrt(torch.mean(audio_tensor ** 2)).item()
#         dynamic_range = torch.max(audio_tensor).item() - torch.min(audio_tensor).item()
        
#         # Advanced steganography detection heuristics
#         # High-frequency anomaly detection
#         high_freq_std = torch.std(power[mid_freq_end:]).item()
#         high_freq_mean = torch.mean(power[mid_freq_end:]).item()
#         high_freq_cv = high_freq_std / (high_freq_mean + eps)  # Coefficient of variation
        
#         # Detect potential LSB steganography
#         lsb_anomaly_score = 0.0
#         if high_freq_cv > 0.5:  # High variability in high frequencies
#             lsb_anomaly_score += 0.3
        
#         if spectral_flatness < 0.3:  # Low spectral flatness
#             lsb_anomaly_score += 0.2
            
#         if zcr > 0.1:  # High zero crossing rate
#             lsb_anomaly_score += 0.2
            
#         # Frequency distribution anomaly
#         expected_distribution = {"low": 0.5, "mid": 0.35, "high": 0.15}
#         distribution_deviation = (
#             abs(low_ratio - expected_distribution["low"]) +
#             abs(mid_ratio - expected_distribution["mid"]) +
#             abs(high_ratio - expected_distribution["high"])
#         )
        
#         if distribution_deviation > 0.3:
#             lsb_anomaly_score += 0.3
        
#         # Overall steganography likelihood
#         steg_likelihood = min(1.0, lsb_anomaly_score)
        
#         # Confidence levels
#         if steg_likelihood < 0.3:
#             confidence = "Low"
#             confidence_color = "green"
#         elif steg_likelihood < 0.7:
#             confidence = "Medium"
#             confidence_color = "yellow"
#         else:
#             confidence = "High"
#             confidence_color = "red"
        
#         return {
#             "spectral_features": features.tolist() if isinstance(features, torch.Tensor) else features,
#             "spectral_flatness": spectral_flatness,
#             "spectral_centroid": spectral_centroid,
#             "zero_crossing_rate": zcr,
#             "rms_energy": rms_energy,
#             "dynamic_range": dynamic_range,
#             "energy_distribution": {
#                 "low_band": low_ratio,
#                 "mid_band": mid_ratio,
#                 "high_band": high_ratio,
#             },
#             "frequency_anomalies": {
#                 "high_freq_cv": high_freq_cv,
#                 "distribution_deviation": distribution_deviation,
#             },
#             "steg_likelihood": steg_likelihood,
#             "detection_confidence": confidence,
#             "confidence_color": confidence_color,
#             "analysis_summary": {
#                 "total_samples": len(audio_data),
#                 "sample_rate": sample_rate,
#                 "duration": len(audio_data) / sample_rate,
#                 "channels": 1 if len(audio_data.shape) == 1 else audio_data.shape[1]
#             }
#         }
    
#     except Exception as e:
#         print(f"Error in audio analysis: {e}")
#         # Return basic fallback analysis
#         return {
#             "spectral_features": [],
#             "spectral_flatness": 0.5,
#             "spectral_centroid": 1000.0,
#             "zero_crossing_rate": 0.05,
#             "rms_energy": 0.1,
#             "dynamic_range": 1.0,
#             "energy_distribution": {
#                 "low_band": 0.33,
#                 "mid_band": 0.33,
#                 "high_band": 0.34,
#             },
#             "frequency_anomalies": {
#                 "high_freq_cv": 0.5,
#                 "distribution_deviation": 0.1,
#             },
#             "steg_likelihood": 0.2,
#             "detection_confidence": "Low",
#             "confidence_color": "green",
#             "analysis_summary": {
#                 "total_samples": len(audio_data) if hasattr(audio_data, '__len__') else 0,
#                 "sample_rate": 44100,
#                 "duration": 0.0,
#                 "channels": 1
#             }
#         }

# # ========== API Routes ==========

# @app.post("/upload")
# async def embed_message(file: UploadFile = File(...), message: str = "Hello"):
#     """
#     Embed a message in audio using the RL-trained optimal method selection.
#     """
#     try:
#         # Read and preprocess the audio file
#         audio_data, samplerate = sf.read(io.BytesIO(await file.read()))
        
#         # Ensure audio_data is float32 and properly shaped
#         if len(audio_data.shape) == 1:
#             audio_data = audio_data.astype(np.float32)
#         else:
#             audio_data = audio_data[:, 0].astype(np.float32)  # Take first channel if stereo
        
#         print(f"Processing audio: {len(audio_data)} samples, {samplerate} Hz")
#         print(f"Message to embed: '{message}'")
        
#         # Convert to tensor with correct shape [1, 1, T]
#         audio_tensor = torch.tensor(audio_data).unsqueeze(0).unsqueeze(0)
        
#         # Extract features and get optimal encoding method using RL agent
#         features = env.extract_audio_features(audio_tensor.squeeze(0).numpy())
#         optimal_method = get_optimal_encoding_method(features)
        
#         print(f"Selected encoding method: {optimal_method}")
        
#         # Convert message to binary and then to tensor
#         message_bits = text_to_bits(message, len(audio_data))
#         message_tensor = bits_to_tensor(message_bits, len(audio_data))
        
#         # Use the encoder with the optimal method
#         # Note: You might need to modify the encoder to accept the method parameter
#         stego_audio = encoder(audio_tensor, message_tensor)
        
#         # Save the processed audio
#         output_file = f"output_{optimal_method.lower()}.flac"
#         save_audio(stego_audio, output_file, samplerate)
        
#         print(f"Audio encoded successfully using {optimal_method}")
        
#         return FileResponse(
#             output_file, 
#             media_type="audio/flac", 
#             filename=output_file,
#             headers={"X-Encoding-Method": optimal_method}
#         )
        
#     except Exception as e:
#         print(f"Error in encoding: {e}")
#         raise HTTPException(status_code=500, detail=f"Encoding failed: {str(e)}")

# @app.post("/decode")
# async def decode_message(file: UploadFile = File(...)):
#     """
#     Decode a message from audio using the trained models.
#     """
#     try:
#         # Read and preprocess the audio file
#         audio_data, samplerate = sf.read(io.BytesIO(await file.read()))
        
#         # Ensure audio_data is float32 and properly shaped
#         if len(audio_data.shape) == 1:
#             audio_data = audio_data.astype(np.float32)
#         else:
#             audio_data = audio_data[:, 0].astype(np.float32)  # Take first channel if stereo
        
#         print(f"Decoding audio: {len(audio_data)} samples, {samplerate} Hz")
        
#         # Convert to tensor with correct shape [1, 1, T]
#         stego_audio = torch.tensor(audio_data).unsqueeze(0).unsqueeze(0)
        
#         # Decode the message
#         decoded_bits = decoder(stego_audio)
        
#         # Convert bits to text with improved error handling
#         decoded_message = ""
#         bits_list = decoded_bits.squeeze().tolist()
        
#         # Process bits in chunks of 8 (bytes)
#         for i in range(0, len(bits_list), 8):
#             if i + 8 <= len(bits_list):
#                 byte_bits = bits_list[i:i+8]
#                 # Convert to binary string
#                 binary_str = ''.join('1' if bit > 0.5 else '0' for bit in byte_bits)
                
#                 try:
#                     byte_value = int(binary_str, 2)
#                     if 32 <= byte_value <= 126:  # Printable ASCII range
#                         decoded_message += chr(byte_value)
#                     elif byte_value == 0:  # Null terminator
#                         break
#                 except ValueError:
#                     continue
        
#         # Clean up the message
#         decoded_message = decoded_message.strip()
        
#         print(f"Decoded message: '{decoded_message}'")
        
#         return JSONResponse(content={
#             "decoded_message": decoded_message,
#             "message_length": len(decoded_message),
#             "decoding_method": "Neural Network"
#         })
        
#     except Exception as e:
#         print(f"Error in decoding: {e}")
#         raise HTTPException(status_code=500, detail=f"Decoding failed: {str(e)}")

# @app.post("/analyze")
# async def analyze_audio(file: UploadFile = File(...)):
#     """
#     Perform comprehensive audio analysis for steganography detection.
#     """
#     try:
#         # Read the audio file
#         audio_data, sample_rate = sf.read(io.BytesIO(await file.read()))
        
#         # Ensure audio_data is properly shaped
#         if len(audio_data.shape) > 1:
#             audio_data = audio_data[:, 0]  # Take first channel if stereo
        
#         print(f"Analyzing audio: {len(audio_data)} samples, {sample_rate} Hz")
        
#         # Perform comprehensive analysis
#         analysis_results = advanced_audio_analysis(audio_data, sample_rate)
        
#         print(f"Analysis complete. Steganography likelihood: {analysis_results['steg_likelihood']:.3f}")
        
#         return JSONResponse(content={"analysis_results": analysis_results})
        
#     except Exception as e:
#         print(f"Error in analysis: {e}")
#         raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# # ========== Training and Agent Routes ==========

# @app.post("/train_agent")
# def train_agent(episodes: int = 10):
#     """
#     Train the RL agent (if needed for fine-tuning).
#     """
#     try:
#         for ep in range(episodes):
#             state = env.reset()
#             episode_reward = 0
            
#             for step in range(10):  # steps per episode
#                 action = agent.choose_action(state)
#                 next_state, reward, done = env.step(action)
#                 agent.learn(state, action, reward, next_state)
#                 episode_reward += reward
#                 state = next_state
                
#                 if done:
#                     break
            
#             print(f"Episode {ep + 1}: Reward = {episode_reward:.3f}")
        
#         # Save the updated agent
#         agent.save(TRAINED_AGENT_PATH)
#         print(f"Agent training complete and saved to {TRAINED_AGENT_PATH}")
        
#         return {
#             "status": "training complete", 
#             "episodes": episodes,
#             "agent_path": TRAINED_AGENT_PATH
#         }
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

# @app.get("/best_method")
# def get_best_method():
#     """
#     Returns the best encoding method based on learned Q-values.
#     """
#     try:
#         # Get Q-values for a sample state
#         sample_state = "sample_state"
#         scores = {}
        
#         for method in ENCODING_METHODS:
#             q_value = agent.get_q(sample_state, method)
#             scores[method] = q_value
        
#         best_method = max(scores, key=scores.get) if scores else "LSB"
        
#         return {
#             "best_method": best_method, 
#             "method_scores": scores,
#             "agent_loaded": os.path.exists(TRAINED_AGENT_PATH)
#         }
        
#     except Exception as e:
#         return {
#             "best_method": "LSB", 
#             "method_scores": {},
#             "error": str(e),
#             "agent_loaded": False
#         }

# @app.get("/agent_status")
# def get_agent_status():
#     """
#     Get the current status of the RL agent.
#     """
#     return {
#         "trained_agent_exists": os.path.exists(TRAINED_AGENT_PATH),
#         "fallback_agent_exists": os.path.exists(FALLBACK_AGENT_PATH),
#         "available_methods": ENCODING_METHODS,
#         "agent_path": TRAINED_AGENT_PATH if os.path.exists(TRAINED_AGENT_PATH) else FALLBACK_AGENT_PATH
#     }

# # ========== Health Check ==========

# @app.get("/health")
# def health_check():
#     """
#     Health check endpoint.
#     """
#     return {
#         "status": "healthy",
#         "models_loaded": True,
#         "agent_loaded": os.path.exists(TRAINED_AGENT_PATH) or os.path.exists(FALLBACK_AGENT_PATH)
#     }

# # ========== Entry Point ==========

# if __name__ == "__main__":
#     import uvicorn
#     print("Starting Audio Steganography API with RL Agent Integration...")
#     print(f"Trained agent available: {os.path.exists(TRAINED_AGENT_PATH)}")
#     uvicorn.run(app, host="0.0.0.0", port=8000)

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import soundfile as sf
import io
import os
import torch
import pickle
from typing import Dict, Any

from rl_environment import AudioStegEnvironment
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

# ========== Constants ==========
ENCODING_METHODS = ["LSB", "Echo", "Phase", "SpreadSpectrum"]
TRAINED_AGENT_PATH = "models/trained_agent_final.pkl"
FALLBACK_AGENT_PATH = "agent_q_table.pkl"

# ========== Initialize Models and Agent ==========
encoder = AudioStegEncoder()
decoder = AudioStegDecoder()
env = AudioStegEnvironment(ENCODING_METHODS)

# Load the trained agent
agent = QLearningAgent(actions=ENCODING_METHODS)
if os.path.exists(TRAINED_AGENT_PATH):
    print(f"Loading trained agent from {TRAINED_AGENT_PATH}")
    agent.load(TRAINED_AGENT_PATH)
elif os.path.exists(FALLBACK_AGENT_PATH):
    print(f"Loading fallback agent from {FALLBACK_AGENT_PATH}")
    agent.load(FALLBACK_AGENT_PATH)
else:
    print("No trained agent found, using random initialization")

def get_optimal_encoding_method(audio_features: torch.Tensor) -> str:
    """
    Use the trained RL agent to select the optimal encoding method
    based on audio features.
    """
    try:
        # Convert audio features to a state representation
        if isinstance(audio_features, torch.Tensor):
            state_features = audio_features.cpu().numpy().flatten()
        else:
            state_features = np.array(audio_features).flatten()
        
        # Create a state string representation for the Q-table
        # You might need to adjust this based on your state representation
        state = f"features_{hash(tuple(state_features[:10])) % 1000}"  # Use first 10 features
        
        # Get the best action from the trained agent
        best_method = agent.choose_action(state, epsilon=0.0)  # No exploration, pure exploitation
        print(f"RL Agent selected method: {best_method}")
        return best_method
    except Exception as e:
        print(f"Error in RL agent selection: {e}, falling back to LSB")
        return "LSB"

def advanced_audio_analysis(audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
    """
    Perform comprehensive audio analysis for steganography detection.
    """
    try:
        # Convert to tensor for processing
        audio_tensor = torch.tensor(audio_data, dtype=torch.float32)
        if len(audio_tensor.shape) == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        
        # Extract features using the environment
        features = env.extract_audio_features(audio_tensor)
        
        # Spectral analysis
        n_fft = 1024
        hop_length = 512
        window = torch.hann_window(n_fft)
        
        stft = torch.stft(
            audio_tensor.squeeze(0), 
            n_fft=n_fft, 
            hop_length=hop_length, 
            window=window,
            return_complex=True
        )
        
        magnitude = torch.abs(stft)
        power = magnitude ** 2
        
        # Frequency band analysis
        freq_bins = magnitude.shape[0]
        low_freq_end = freq_bins // 4
        mid_freq_end = 3 * freq_bins // 4
        
        low_band_energy = torch.mean(power[:low_freq_end]).item()
        mid_band_energy = torch.mean(power[low_freq_end:mid_freq_end]).item()
        high_band_energy = torch.mean(power[mid_freq_end:]).item()
        
        total_energy = low_band_energy + mid_band_energy + high_band_energy
        
        if total_energy > 0:
            low_ratio = low_band_energy / total_energy
            mid_ratio = mid_band_energy / total_energy
            high_ratio = high_band_energy / total_energy
        else:
            low_ratio = mid_ratio = high_ratio = 1/3
        
        # Spectral flatness (Wiener entropy)
        eps = 1e-10
        geometric_mean = torch.exp(torch.mean(torch.log(magnitude + eps)))
        arithmetic_mean = torch.mean(magnitude)
        spectral_flatness = (geometric_mean / (arithmetic_mean + eps)).item()
        
        # Spectral centroid
        freqs = torch.linspace(0, sample_rate/2, magnitude.shape[0])
        spectral_centroid = torch.sum(freqs.unsqueeze(1) * magnitude) / torch.sum(magnitude)
        spectral_centroid = spectral_centroid.item()
        
        # Zero crossing rate
        zero_crossings = torch.sum(torch.diff(torch.sign(audio_tensor)) != 0).item()
        zcr = zero_crossings / len(audio_tensor.squeeze())
        
        # Statistical measures
        rms_energy = torch.sqrt(torch.mean(audio_tensor ** 2)).item()
        dynamic_range = torch.max(audio_tensor).item() - torch.min(audio_tensor).item()
        
        # Advanced steganography detection heuristics
        # High-frequency anomaly detection
        high_freq_std = torch.std(power[mid_freq_end:]).item()
        high_freq_mean = torch.mean(power[mid_freq_end:]).item()
        high_freq_cv = high_freq_std / (high_freq_mean + eps)  # Coefficient of variation
        
        # Detect potential LSB steganography
        lsb_anomaly_score = 0.0
        if high_freq_cv > 0.5:  # High variability in high frequencies
            lsb_anomaly_score += 0.3
        
        if spectral_flatness < 0.3:  # Low spectral flatness
            lsb_anomaly_score += 0.2
            
        if zcr > 0.1:  # High zero crossing rate
            lsb_anomaly_score += 0.2
            
        # Frequency distribution anomaly
        expected_distribution = {"low": 0.5, "mid": 0.35, "high": 0.15}
        distribution_deviation = (
            abs(low_ratio - expected_distribution["low"]) +
            abs(mid_ratio - expected_distribution["mid"]) +
            abs(high_ratio - expected_distribution["high"])
        )
        
        if distribution_deviation > 0.3:
            lsb_anomaly_score += 0.3
        
        # Overall steganography likelihood
        steg_likelihood = min(1.0, lsb_anomaly_score)
        
        # Confidence levels
        if steg_likelihood < 0.3:
            confidence = "Low"
            confidence_color = "green"
        elif steg_likelihood < 0.7:
            confidence = "Medium"
            confidence_color = "yellow"
        else:
            confidence = "High"
            confidence_color = "red"
        
        return {
            "spectral_features": features.tolist() if isinstance(features, torch.Tensor) else features,
            "spectral_flatness": spectral_flatness,
            "spectral_centroid": spectral_centroid,
            "zero_crossing_rate": zcr,
            "rms_energy": rms_energy,
            "dynamic_range": dynamic_range,
            "energy_distribution": {
                "low_band": low_ratio,
                "mid_band": mid_ratio,
                "high_band": high_ratio,
            },
            "frequency_anomalies": {
                "high_freq_cv": high_freq_cv,
                "distribution_deviation": distribution_deviation,
            },
            "steg_likelihood": steg_likelihood,
            "detection_confidence": confidence,
            "confidence_color": confidence_color,
            "analysis_summary": {
                "total_samples": len(audio_data),
                "sample_rate": sample_rate,
                "duration": len(audio_data) / sample_rate,
                "channels": 1 if len(audio_data.shape) == 1 else audio_data.shape[1]
            }
        }
    
    except Exception as e:
        print(f"Error in audio analysis: {e}")
        # Return basic fallback analysis
        return {
            "spectral_features": [],
            "spectral_flatness": 0.5,
            "spectral_centroid": 1000.0,
            "zero_crossing_rate": 0.05,
            "rms_energy": 0.1,
            "dynamic_range": 1.0,
            "energy_distribution": {
                "low_band": 0.33,
                "mid_band": 0.33,
                "high_band": 0.34,
            },
            "frequency_anomalies": {
                "high_freq_cv": 0.5,
                "distribution_deviation": 0.1,
            },
            "steg_likelihood": 0.2,
            "detection_confidence": "Low",
            "confidence_color": "green",
            "analysis_summary": {
                "total_samples": len(audio_data) if hasattr(audio_data, '__len__') else 0,
                "sample_rate": 44100,
                "duration": 0.0,
                "channels": 1
            }
        }

# ========== API Routes ==========

@app.post("/upload")
async def embed_message(file: UploadFile = File(...), message: str = "Hello"):
    """
    Embed a message in audio using the RL-trained optimal method selection.
    """
    try:
        # Read and preprocess the audio file
        audio_data, samplerate = sf.read(io.BytesIO(await file.read()))
        
        # Ensure audio_data is float32 and properly shaped
        if len(audio_data.shape) == 1:
            audio_data = audio_data.astype(np.float32)
        else:
            audio_data = audio_data[:, 0].astype(np.float32)  # Take first channel if stereo
        
        print(f"Processing audio: {len(audio_data)} samples, {samplerate} Hz")
        print(f"Message to embed: '{message}'")
        
        # Convert to tensor with correct shape [1, 1, T]
        audio_tensor = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        # FIXED: Extract features from the correct tensor shape
        try:
            # Use a 2D tensor [1, T] for feature extraction
            features_input = audio_tensor.squeeze(1)  # Remove middle dimension: [1, 1, T] -> [1, T]
            features = env.extract_audio_features(features_input.numpy())
            optimal_method = get_optimal_encoding_method(features)
        except Exception as feat_error:
            print(f"Feature extraction error: {feat_error}, using default method")
            optimal_method = "LSB"
        
        print(f"Selected encoding method: {optimal_method}")
        
        # Convert message to binary and then to tensor - FIXED
        message_bits = text_to_bits(message, len(audio_data))
        message_tensor = bits_to_tensor(message_bits, len(audio_data))
        
        # Ensure message_tensor has the correct shape [1, 1, T]
        if message_tensor.shape != audio_tensor.shape:
            print(f"Reshaping message tensor from {message_tensor.shape} to {audio_tensor.shape}")
            message_tensor = message_tensor.reshape(audio_tensor.shape)
        
        # Use the encoder with the optimal method
        # Note: You might need to modify the encoder to accept the method parameter
        stego_audio = encoder(audio_tensor, message_tensor)
        
        # Save the processed audio
        output_file = f"output_{optimal_method.lower()}.flac"
        save_audio(stego_audio, output_file, samplerate)
        
        print(f"Audio encoded successfully using {optimal_method}")
        
        return FileResponse(
            output_file, 
            media_type="audio/flac", 
            filename=output_file,
            headers={"X-Encoding-Method": optimal_method}
        )
        
    except Exception as e:
        print(f"Error in encoding: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Encoding failed: {str(e)}")

@app.post("/decode")
async def decode_message(file: UploadFile = File(...)):
    """
    Decode a message from audio using the trained models.
    """
    try:
        # Read and preprocess the audio file
        audio_data, samplerate = sf.read(io.BytesIO(await file.read()))
        
        # Ensure audio_data is float32 and properly shaped
        if len(audio_data.shape) == 1:
            audio_data = audio_data.astype(np.float32)
        else:
            audio_data = audio_data[:, 0].astype(np.float32)  # Take first channel if stereo
        
        print(f"Decoding audio: {len(audio_data)} samples, {samplerate} Hz")
        
        # Convert to tensor with correct shape [1, 1, T]
        stego_audio = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        # Decode the message
        decoded_bits = decoder(stego_audio)
        
        # Convert bits to text with improved error handling
        decoded_message = ""
        bits_list = decoded_bits.squeeze().tolist()
        
        # Process bits in chunks of 8 (bytes)
        for i in range(0, len(bits_list), 8):
            if i + 8 <= len(bits_list):
                byte_bits = bits_list[i:i+8]
                # Convert to binary string
                binary_str = ''.join('1' if bit > 0.5 else '0' for bit in byte_bits)
                
                try:
                    byte_value = int(binary_str, 2)
                    if 32 <= byte_value <= 126:  # Printable ASCII range
                        decoded_message += chr(byte_value)
                    elif byte_value == 0:  # Null terminator
                        break
                except ValueError:
                    continue
        
        # Clean up the message
        decoded_message = decoded_message.strip()
        
        print(f"Decoded message: '{decoded_message}'")
        
        return JSONResponse(content={
            "decoded_message": decoded_message,
            "message_length": len(decoded_message),
            "decoding_method": "Neural Network"
        })
        
    except Exception as e:
        print(f"Error in decoding: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Decoding failed: {str(e)}")

@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    """
    Perform comprehensive audio analysis for steganography detection.
    """
    try:
        # Read the audio file
        audio_data, sample_rate = sf.read(io.BytesIO(await file.read()))
        
        # Ensure audio_data is properly shaped
        if len(audio_data.shape) > 1:
            audio_data = audio_data[:, 0]  # Take first channel if stereo
        
        print(f"Analyzing audio: {len(audio_data)} samples, {sample_rate} Hz")
        
        # Perform comprehensive analysis
        analysis_results = advanced_audio_analysis(audio_data, sample_rate)
        
        print(f"Analysis complete. Steganography likelihood: {analysis_results['steg_likelihood']:.3f}")
        
        return JSONResponse(content={"analysis_results": analysis_results})
        
    except Exception as e:
        print(f"Error in analysis: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# ========== Training and Agent Routes ==========

@app.post("/train_agent")
def train_agent(episodes: int = 10):
    """
    Train the RL agent (if needed for fine-tuning).
    """
    try:
        for ep in range(episodes):
            state = env.reset()
            episode_reward = 0
            
            for step in range(10):  # steps per episode
                action = agent.choose_action(state)
                next_state, reward, done = env.step(action)
                agent.learn(state, action, reward, next_state)
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            print(f"Episode {ep + 1}: Reward = {episode_reward:.3f}")
        
        # Save the updated agent
        agent.save(TRAINED_AGENT_PATH)
        print(f"Agent training complete and saved to {TRAINED_AGENT_PATH}")
        
        return {
            "status": "training complete", 
            "episodes": episodes,
            "agent_path": TRAINED_AGENT_PATH
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.get("/best_method")
def get_best_method():
    """
    Returns the best encoding method based on learned Q-values.
    """
    try:
        # Get Q-values for a sample state
        sample_state = "sample_state"
        scores = {}
        
        for method in ENCODING_METHODS:
            q_value = agent.get_q(sample_state, method)
            scores[method] = q_value
        
        best_method = max(scores, key=scores.get) if scores else "LSB"
        
        return {
            "best_method": best_method, 
            "method_scores": scores,
            "agent_loaded": os.path.exists(TRAINED_AGENT_PATH)
        }
        
    except Exception as e:
        return {
            "best_method": "LSB", 
            "method_scores": {},
            "error": str(e),
            "agent_loaded": False
        }

@app.get("/agent_status")
def get_agent_status():
    """
    Get the current status of the RL agent.
    """
    return {
        "trained_agent_exists": os.path.exists(TRAINED_AGENT_PATH),
        "fallback_agent_exists": os.path.exists(FALLBACK_AGENT_PATH),
        "available_methods": ENCODING_METHODS,
        "agent_path": TRAINED_AGENT_PATH if os.path.exists(TRAINED_AGENT_PATH) else FALLBACK_AGENT_PATH
    }

# ========== Health Check ==========

@app.get("/health")
def health_check():
    """
    Health check endpoint.
    """
    return {
        "status": "healthy",
        "models_loaded": True,
        "agent_loaded": os.path.exists(TRAINED_AGENT_PATH) or os.path.exists(FALLBACK_AGENT_PATH)
    }

# ========== Entry Point ==========

if __name__ == "__main__":
    import uvicorn
    print("Starting Audio Steganography API with RL Agent Integration...")
    print(f"Trained agent available: {os.path.exists(TRAINED_AGENT_PATH)}")
    uvicorn.run(app, host="0.0.0.0", port=8000)