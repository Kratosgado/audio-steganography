from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
import numpy as np
import soundfile as sf
import io
import os
import torch
import pickle
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from core_modules.framework import RLAudioSteganography

try:
    import scipy.stats
except ImportError:
    print(
        "Warning: scipy not available, some statistical analysis features will be limited"
    )
    scipy = None
from utils import simple_lsb_embed, simple_lsb_extract, save_audio

from rl_environment import AudioStegEnvironment
from rl_agent import QLearningAgent, DeepQLearningAgent, RLSteganographyManager
from encoder import AudioStegEncoder, AdvancedAudioStegEncoder
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
TRAINED_AGENT_PATH = "models/multi_message_agent_final.pth"
FALLBACK_AGENT_PATH = "models/trained_agent_final.pth"

# ========== Initialize Models and Agent ==========
# Initialize components
encoding_methods = ["neural", "lsb", "spread_spectrum", "echo_hiding", "phase_coding"]

# Initialize encoders and decoder
encoder = AudioStegEncoder()  # Legacy encoder
advanced_encoder = AdvancedAudioStegEncoder()  # New advanced encoder
decoder = AudioStegDecoder()

# Initialize RL environment
rl_env = AudioStegEnvironment()

# Initialize RL Steganography Manager with Deep RL
rl_manager = RLSteganographyManager(use_deep_rl=True)

# Try to load trained RL agent
model_path = "models/deep_rl_agent.pth"
try:
    if rl_manager.load_agent(model_path):
        print("Loaded trained Deep RL agent")
    else:
        print("Initialized new Deep RL agent")
except Exception as e:
    print(f"Error loading RL agent: {e}. Using new agent.")

# Fallback to tabular Q-learning if needed
try:
    with open("models/rl_agent.pkl", "rb") as f:
        fallback_agent = pickle.load(f)
    print("Loaded fallback tabular RL agent")
except:
    fallback_agent = QLearningAgent(actions=encoding_methods)
    print("Initialized fallback tabular RL agent")
# Legacy agent loading for backward compatibility
legacy_agent_path = "models/trained_agent.pkl"
if os.path.exists(legacy_agent_path):
    print(f"Loading legacy trained agent from {legacy_agent_path}")
    try:
        fallback_agent.load(legacy_agent_path)
    except Exception as e:
        print(f"Error loading legacy agent: {e}")
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
        best_method = fallback_agent.choose_action(
            state, epsilon=0.0
        )  # No exploration, pure exploitation
        print(f"RL Agent selected method: {best_method}")
        return best_method
    except Exception as e:
        print(f"Error in RL agent selection: {e}, falling back to LSB")
        return "LSB"


def enhanced_audio_analysis(
    audio_data: np.ndarray, sample_rate: int, filename: str = None
) -> Dict[str, Any]:
    """
    Enhanced comprehensive audio analysis for steganography detection.
    Includes advanced ML-based detection and forensic analysis.
    """
    try:
        # Basic audio analysis first
        basic_analysis = advanced_audio_analysis(audio_data, sample_rate)

        # Enhanced steganography detection
        enhanced_detection = perform_enhanced_steg_detection(audio_data, sample_rate)

        # File metadata analysis
        metadata_analysis = analyze_file_metadata(filename) if filename else {}

        # Combine all analyses
        enhanced_results = {
            **basic_analysis,
            "enhanced_detection": enhanced_detection,
            "metadata_analysis": metadata_analysis,
            "analysis_version": "2.0",
            "analysis_timestamp": datetime.now().isoformat(),
        }

        return enhanced_results

    except Exception as e:
        print(f"Enhanced analysis failed: {e}, falling back to basic analysis")
        return advanced_audio_analysis(audio_data, sample_rate)


def perform_enhanced_steg_detection(
    audio_data: np.ndarray, sample_rate: int
) -> Dict[str, Any]:
    """
    Perform enhanced steganography detection using advanced techniques.
    """
    try:
        # LSB analysis
        lsb_analysis = analyze_lsb_patterns(audio_data)

        # Frequency domain analysis
        freq_analysis = analyze_frequency_anomalies(audio_data, sample_rate)

        # Statistical analysis
        stat_analysis = analyze_statistical_anomalies(audio_data)

        # Entropy analysis
        entropy_analysis = analyze_entropy_patterns(audio_data)

        return {
            "lsb_analysis": lsb_analysis,
            "frequency_analysis": freq_analysis,
            "statistical_analysis": stat_analysis,
            "entropy_analysis": entropy_analysis,
        }
    except Exception as e:
        print(f"Enhanced detection failed: {e}")
        return {}


def analyze_lsb_patterns(audio_data: np.ndarray) -> Dict[str, Any]:
    """
    Analyze LSB patterns for steganography detection.
    """
    try:
        # Convert to 16-bit integers for LSB analysis
        audio_16bit = (audio_data * 32767).astype(np.int16)

        # Extract LSBs
        lsbs = audio_16bit & 1

        # Calculate LSB statistics
        lsb_mean = np.mean(lsbs)
        lsb_std = np.std(lsbs)

        # Chi-square test for randomness
        ones = np.sum(lsbs)
        zeros = len(lsbs) - ones
        expected = len(lsbs) / 2
        chi_square = ((ones - expected) ** 2 + (zeros - expected) ** 2) / expected

        # Runs test for randomness
        runs = 1
        for i in range(1, len(lsbs)):
            if lsbs[i] != lsbs[i - 1]:
                runs += 1

        expected_runs = (2 * ones * zeros) / len(lsbs) + 1
        runs_anomaly = (
            abs(runs - expected_runs) / expected_runs if expected_runs > 0 else 0
        )

        return {
            "lsb_mean": float(lsb_mean),
            "lsb_std": float(lsb_std),
            "chi_square": float(chi_square),
            "runs_test": float(runs_anomaly),
            "lsb_anomaly_score": min(1.0, (chi_square / 10 + runs_anomaly) / 2),
        }
    except Exception as e:
        print(f"LSB analysis failed: {e}")
        return {"lsb_anomaly_score": 0.0}


def analyze_frequency_anomalies(
    audio_data: np.ndarray, sample_rate: int
) -> Dict[str, Any]:
    """
    Analyze frequency domain for steganography artifacts.
    """
    try:
        # FFT analysis
        fft = np.fft.fft(audio_data)
        magnitude = np.abs(fft)

        # High frequency analysis
        high_freq_start = len(magnitude) // 2
        high_freq_energy = np.mean(magnitude[high_freq_start:])
        total_energy = np.mean(magnitude)

        high_freq_ratio = high_freq_energy / total_energy if total_energy > 0 else 0

        # Spectral irregularities
        spectral_diff = np.diff(magnitude)
        spectral_variance = np.var(spectral_diff)

        return {
            "high_freq_ratio": float(high_freq_ratio),
            "spectral_variance": float(spectral_variance),
            "freq_anomaly_score": min(
                1.0, high_freq_ratio * 2 + spectral_variance / 1000
            ),
        }
    except Exception as e:
        print(f"Frequency analysis failed: {e}")
        return {"freq_anomaly_score": 0.0}


def analyze_statistical_anomalies(audio_data: np.ndarray) -> Dict[str, Any]:
    """
    Analyze statistical properties for anomalies.
    """
    try:
        # Basic statistics
        mean = np.mean(audio_data)
        std = np.std(audio_data)
        skewness = float(scipy.stats.skew(audio_data)) if "scipy" in globals() else 0.0
        kurtosis = (
            float(scipy.stats.kurtosis(audio_data)) if "scipy" in globals() else 0.0
        )

        # Histogram analysis
        hist, _ = np.histogram(audio_data, bins=50)
        hist_variance = np.var(hist)

        return {
            "mean": float(mean),
            "std": float(std),
            "skewness": skewness,
            "kurtosis": kurtosis,
            "histogram_variance": float(hist_variance),
            "stat_anomaly_score": min(1.0, abs(skewness) / 5 + abs(kurtosis) / 10),
        }
    except Exception as e:
        print(f"Statistical analysis failed: {e}")
        return {"stat_anomaly_score": 0.0}


def analyze_entropy_patterns(audio_data: np.ndarray) -> Dict[str, Any]:
    """
    Analyze entropy patterns for steganography detection.
    """
    try:
        # Convert to discrete values for entropy calculation
        audio_discrete = (audio_data * 1000).astype(int)

        # Calculate entropy
        _, counts = np.unique(audio_discrete, return_counts=True)
        probabilities = counts / len(audio_discrete)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))

        # Local entropy analysis
        chunk_size = len(audio_data) // 10
        local_entropies = []

        for i in range(0, len(audio_data) - chunk_size, chunk_size):
            chunk = audio_data[i : i + chunk_size]
            chunk_discrete = (chunk * 1000).astype(int)
            _, chunk_counts = np.unique(chunk_discrete, return_counts=True)
            chunk_probs = chunk_counts / len(chunk_discrete)
            chunk_entropy = -np.sum(chunk_probs * np.log2(chunk_probs + 1e-10))
            local_entropies.append(chunk_entropy)

        entropy_variance = np.var(local_entropies) if local_entropies else 0

        return {
            "global_entropy": float(entropy),
            "entropy_variance": float(entropy_variance),
            "entropy_anomaly_score": min(1.0, entropy_variance / 10),
        }
    except Exception as e:
        print(f"Entropy analysis failed: {e}")
        return {"entropy_anomaly_score": 0.0}


def analyze_file_metadata(filename: str) -> Dict[str, Any]:
    """
    Analyze file metadata for suspicious patterns.
    """
    try:
        if not filename:
            return {}

        # File extension analysis
        ext = filename.lower().split(".")[-1] if "." in filename else ""

        # Suspicious patterns
        suspicious_patterns = ["stego", "hidden", "secret", "encoded", "embed"]
        has_suspicious_name = any(
            pattern in filename.lower() for pattern in suspicious_patterns
        )

        return {
            "filename": filename,
            "extension": ext,
            "has_suspicious_name": has_suspicious_name,
            "metadata_risk_score": 0.3 if has_suspicious_name else 0.0,
        }
    except Exception as e:
        print(f"Metadata analysis failed: {e}")
        return {}


def calculate_enhanced_likelihood(
    basic_analysis: Dict, enhanced_detection: Dict, metadata_analysis: Dict
) -> float:
    """
    Calculate enhanced steganography likelihood score.
    """
    try:
        # Base likelihood from basic analysis
        base_likelihood = basic_analysis.get("steg_likelihood", 0.0)

        # Enhanced detection scores
        lsb_score = enhanced_detection.get("lsb_analysis", {}).get(
            "lsb_anomaly_score", 0.0
        )
        freq_score = enhanced_detection.get("frequency_analysis", {}).get(
            "freq_anomaly_score", 0.0
        )
        stat_score = enhanced_detection.get("statistical_analysis", {}).get(
            "stat_anomaly_score", 0.0
        )
        entropy_score = enhanced_detection.get("entropy_analysis", {}).get(
            "entropy_anomaly_score", 0.0
        )

        # Metadata risk
        metadata_score = metadata_analysis.get("metadata_risk_score", 0.0)

        # Weighted combination
        enhanced_likelihood = (
            base_likelihood * 0.3
            + lsb_score * 0.25
            + freq_score * 0.2
            + stat_score * 0.15
            + entropy_score * 0.1
            + metadata_score * 0.1
        )

        return min(1.0, enhanced_likelihood)
    except Exception as e:
        print(f"Enhanced likelihood calculation failed: {e}")
        return basic_analysis.get("steg_likelihood", 0.0)


def get_rl_steganography_assessment(audio_features: torch.Tensor) -> Dict[str, Any]:
    """
    Use RL agent to assess steganography likelihood.
    """
    try:
        # Use the RL manager to get encoding method recommendation
        encoding_method = get_optimal_encoding_method(audio_features)

        # Simulate RL-based steganography assessment
        # In a real implementation, this would use a trained classifier
        feature_variance = torch.var(audio_features).item()
        feature_mean = torch.mean(audio_features).item()

        # Simple heuristic based on feature characteristics
        rl_likelihood = min(
            1.0, abs(feature_variance - 0.1) * 2 + abs(feature_mean) * 0.5
        )

        return {
            "recommended_method": encoding_method,
            "feature_variance": feature_variance,
            "feature_mean": feature_mean,
            "rl_likelihood": rl_likelihood,
            "confidence": "medium" if rl_likelihood > 0.5 else "low",
        }
    except Exception as e:
        print(f"RL assessment failed: {e}")
        return {"rl_likelihood": 0.0, "confidence": "low"}

        # Recalculate overall likelihood with enhanced features
        enhanced_results["steg_likelihood"] = calculate_enhanced_likelihood(
            basic_analysis, enhanced_detection, metadata_analysis
        )

        # Update confidence based on enhanced analysis
        likelihood = enhanced_results["steg_likelihood"]
        if likelihood < 0.25:
            enhanced_results["detection_confidence"] = "Very Low"
            enhanced_results["confidence_color"] = "green"
        elif likelihood < 0.5:
            enhanced_results["detection_confidence"] = "Low"
            enhanced_results["confidence_color"] = "green"
        elif likelihood < 0.75:
            enhanced_results["detection_confidence"] = "Medium"
            enhanced_results["confidence_color"] = "yellow"
        else:
            enhanced_results["detection_confidence"] = "High"
            enhanced_results["confidence_color"] = "red"

        return enhanced_results

    except Exception as e:
        print(f"Enhanced analysis failed, falling back to basic: {e}")
        return advanced_audio_analysis(audio_data, sample_rate)


def perform_message_comparison_analysis(
    audio_data: np.ndarray, sample_rate: int, original_message: str
) -> Dict[str, Any]:
    """
    Perform message decoding and comparison analysis.
    """
    try:
        print(f"Performing message comparison analysis for: '{original_message}'")

        # Initialize RL environment with the audio
        rl_env.original_audio = audio_data
        rl_env.current_audio = audio_data.copy()
        rl_env.sample_rate = sample_rate

        # Try multiple decoding methods
        decoded_message = ""
        decoding_method = "Unknown"
        decoding_confidence = 0.0

        # Method 1: Simple LSB extraction
        try:
            decoded_message = simple_lsb_extract(audio_data)
            if decoded_message:
                decoding_method = "Simple LSB"
                decoding_confidence = 0.8
                print(f"Simple LSB extracted: '{decoded_message}'")
        except Exception as e:
            print(f"Simple LSB extraction failed: {e}")

        # Method 2: RL environment decoding if simple LSB failed
        if not decoded_message:
            try:
                print("Trying RL environment decoding...")
                possible_lengths = [
                    len(original_message) * 8,
                    40,
                    48,
                    56,
                    64,
                    80,
                    96,
                    120,
                    160,
                    200,
                    240,
                    320,
                    400,
                ]

                for bit_length in possible_lengths:
                    try:
                        decoded_bits = rl_env.decode_message(bit_length)

                        # Convert bits to text
                        chars = []
                        for i in range(0, len(decoded_bits), 8):
                            if i + 8 <= len(decoded_bits):
                                byte = decoded_bits[i : i + 8]
                                char_code = int(byte, 2)
                                if 32 <= char_code <= 126:  # Printable ASCII
                                    chars.append(chr(char_code))
                                else:
                                    break

                        candidate_message = "".join(chars)

                        # Check if this looks like a valid message
                        if len(candidate_message) > 0 and all(
                            32 <= ord(c) <= 126 for c in candidate_message
                        ):
                            decoded_message = candidate_message
                            decoding_method = "RL-Enhanced LSB"
                            decoding_confidence = 0.6
                            print(f"RL extracted: '{decoded_message}'")
                            break

                    except Exception:
                        continue

            except Exception as e:
                print(f"RL environment decoding failed: {e}")

        # Compare messages
        messages_match = False
        similarity_score = 0.0

        if decoded_message:
            # Exact match
            if decoded_message.strip().lower() == original_message.strip().lower():
                messages_match = True
                similarity_score = 1.0
            else:
                # Calculate similarity score
                from difflib import SequenceMatcher

                similarity_score = SequenceMatcher(
                    None, decoded_message.lower(), original_message.lower()
                ).ratio()

                # Consider it a match if similarity is high enough
                if similarity_score > 0.8:
                    messages_match = True

        # Calculate message analysis confidence
        analysis_confidence = "Low"
        if messages_match:
            analysis_confidence = "Very High"
        elif decoded_message and similarity_score > 0.5:
            analysis_confidence = "High"
        elif decoded_message:
            analysis_confidence = "Medium"

        return {
            "original_message": original_message,
            "decoded_message": decoded_message,
            "messages_match": messages_match,
            "similarity_score": similarity_score,
            "decoding_method": decoding_method,
            "decoding_confidence": decoding_confidence,
            "analysis_confidence": analysis_confidence,
            "message_length_original": len(original_message),
            "message_length_decoded": len(decoded_message) if decoded_message else 0,
        }

    except Exception as e:
        print(f"Message comparison analysis failed: {e}")
        return {
            "original_message": original_message,
            "decoded_message": None,
            "messages_match": False,
            "similarity_score": 0.0,
            "decoding_method": "Error",
            "decoding_confidence": 0.0,
            "analysis_confidence": "Error",
            "error": str(e),
        }


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
        features = rl_env.extract_audio_features(audio_tensor)

        # Spectral analysis
        n_fft = 1024
        hop_length = 512
        window = torch.hann_window(n_fft)

        stft = torch.stft(
            audio_tensor.squeeze(0),
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            return_complex=True,
        )

        magnitude = torch.abs(stft)
        power = magnitude**2

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
            low_ratio = mid_ratio = high_ratio = 1 / 3

        # Spectral flatness (Wiener entropy)
        eps = 1e-10
        geometric_mean = torch.exp(torch.mean(torch.log(magnitude + eps)))
        arithmetic_mean = torch.mean(magnitude)
        spectral_flatness = (geometric_mean / (arithmetic_mean + eps)).item()

        # Spectral centroid
        freqs = torch.linspace(0, sample_rate / 2, magnitude.shape[0])
        spectral_centroid = torch.sum(freqs.unsqueeze(1) * magnitude) / torch.sum(
            magnitude
        )
        spectral_centroid = spectral_centroid.item()

        # Zero crossing rate
        zero_crossings = torch.sum(torch.diff(torch.sign(audio_tensor)) != 0).item()
        zcr = zero_crossings / len(audio_tensor.squeeze())

        # Statistical measures
        rms_energy = torch.sqrt(torch.mean(audio_tensor**2)).item()
        dynamic_range = torch.max(audio_tensor).item() - torch.min(audio_tensor).item()

        # Advanced steganography detection heuristics
        # High-frequency anomaly detection
        high_freq_std = torch.std(power[mid_freq_end:]).item()
        high_freq_mean = torch.mean(power[mid_freq_end:]).item()
        high_freq_cv = high_freq_std / (
            high_freq_mean + eps
        )  # Coefficient of variation

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
            abs(low_ratio - expected_distribution["low"])
            + abs(mid_ratio - expected_distribution["mid"])
            + abs(high_ratio - expected_distribution["high"])
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
            "spectral_features": (
                features.tolist() if isinstance(features, torch.Tensor) else features
            ),
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
                "channels": 1 if len(audio_data.shape) == 1 else audio_data.shape[1],
            },
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
                "total_samples": (
                    len(audio_data) if hasattr(audio_data, "__len__") else 0
                ),
                "sample_rate": 44100,
                "duration": 0.0,
                "channels": 1,
            },
        }


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


@app.get("/")
async def read_root():
    return {"message": "Welcome to the audio steganography API!"}


@app.post("/upload")
async def embed_message(file: UploadFile = File(...), message: str = "Hello"):
    """
    Embed a message in audio using the trained RL agent for optimal method selection.
    """
    try:
        framework = RLAudioSteganography()
        audio_data, samplerate = sf.read(io.BytesIO(await file.read()))
        framework.Initialize_components(audio_data, method="spread-spectrum")
        # Read and preprocess the audio file
        

        # Ensure audio_data is float32 and properly shaped
        if len(audio_data.shape) == 1:
            audio_data = audio_data.astype(np.float32)
        else:
            audio_data = audio_data[:, 0].astype(
                np.float32
            )  # Take first channel if stereo

        print(f"Processing audio: {len(audio_data)} samples, {samplerate} Hz")
        print(f"Message to embed: '{message}'")

        # # Initialize RL environment with the audio
        # rl_env.original_audio = audio_data
        # rl_env.current_audio = audio_data.copy()
        # rl_env.sample_rate = samplerate

        # Use simple LSB embedding for now (since it works with simple extraction)
        try:
            # Get optimal encoding method using RL agent
            # audio_features = rl_env.extract_audio_features(audio_data)
            # optimal_method = rl_manager.select_encoding_method(audio_features)
            # print(f"RL Agent selected method: {optimal_method}")

            # Use simple LSB embedding for compatibility
            # stego_audio = simple_lsb_embed(audio_data, message)
            stego_audio = framework.embed_message(audio_data, message)

        except Exception as rl_error:
            print(f"RL agent error: {rl_error}, falling back to LSB")
            optimal_method = "LSB"
            stego_audio = simple_lsb_embed(audio_data, message)

        # Save the processed audio as WAV to preserve LSBs
        output_file = f"output_{optimal_method.lower()}.wav"
        save_audio(stego_audio, output_file, samplerate)

        print(f"Audio encoded successfully using {optimal_method}")

        return FileResponse(
            output_file,
            media_type="audio/wav",
            filename=output_file,
            headers={"X-Encoding-Method": optimal_method},
        )

    except Exception as e:
        print(f"Error in encoding: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Encoding failed: {str(e)}")


@app.post("/decode")
async def decode_message(file: UploadFile = File(...)):
    """
    Decode a message from audio using the trained RL environment.
    """
    try:
        # Read and preprocess the audio file
        audio_data, samplerate = sf.read(io.BytesIO(await file.read()))
        framework = RLAudioSteganography()
        framework.Initialize_components(audio_data, method="spread-spectrum")

        # Ensure audio_data is float32 and properly shaped
        if len(audio_data.shape) == 1:
            audio_data = audio_data.astype(np.float32)
        else:
            audio_data = audio_data[:, 0].astype(
                np.float32
            )  # Take first channel if stereo

        print(f"Decoding audio: {len(audio_data)} samples, {samplerate} Hz")

        # Initialize RL environment with the audio
        # rl_env.original_audio = audio_data
        # rl_env.current_audio = audio_data.copy()
        # rl_env.sample_rate = samplerate

        # Use simple LSB extraction (which we know works with our encoding)
        decoded_message = ""
        decoding_method = "Unknown"

        try:
            # decoded_message = simple_lsb_extract(audio_data)
            decoded_message = framework.extract_message(audio_data, 4)
            print(f"Simple LSB extracted: '{decoded_message}'")

            if decoded_message:
                decoding_method = "Spread Spectrum"
            else:
                print("Simple LSB failed, trying RL environment...")
                # Try to decode with different message lengths
                possible_lengths = [
                    40,
                    48,
                    56,
                    64,
                    80,
                    96,
                    120,
                    160,
                    200,
                    240,
                    320,
                    400,
                ]

                for bit_length in possible_lengths:
                    try:
                        decoded_bits = rl_env.decode_message(bit_length)

                        # Convert bits to text
                        chars = []
                        for i in range(0, len(decoded_bits), 8):
                            if i + 8 <= len(decoded_bits):
                                byte = decoded_bits[i : i + 8]
                                char_code = int(byte, 2)
                                if 32 <= char_code <= 126:  # Printable ASCII
                                    chars.append(chr(char_code))
                                else:
                                    break

                        candidate_message = "".join(chars)

                        # Check if this looks like a valid message
                        if len(candidate_message) > 0 and all(
                            32 <= ord(c) <= 126 for c in candidate_message
                        ):
                            if any(
                                word in candidate_message.lower()
                                for word in ["hello", "test", "secret", "message"]
                            ):
                                decoded_message = candidate_message
                                decoding_method = "RL-Enhanced LSB"
                                break
                            elif len(candidate_message) >= 3 and not decoded_message:
                                decoded_message = candidate_message
                                decoding_method = "RL-Enhanced LSB"

                    except Exception:
                        continue

        except Exception as decode_error:
            print(f"Decode error: {decode_error}")
            decoded_message = ""
            decoding_method = "Error"

        print(f"Decoded message: '{decoded_message}' using {decoding_method}")

        return JSONResponse(
            content={
                "decoded_message": decoded_message,
                "message_length": len(decoded_message),
                "decoding_method": decoding_method,
            }
        )

    except Exception as e:
        print(f"Error in decoding: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Decoding failed: {str(e)}")


@app.post("/analyze")
async def analyze_audio(
    file: UploadFile = File(...), original_message: str = Form(None)
):
    """
    Perform comprehensive audio analysis for steganography detection.
    Enhanced with ML-based detection, detailed forensic analysis, and message comparison.
    """
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith("audio/"):
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Please upload an audio file.",
            )

        # Read the audio file
        file_content = await file.read()
        audio_data, sample_rate = sf.read(io.BytesIO(file_content))

        # Ensure audio_data is properly shaped
        if len(audio_data.shape) > 1:
            audio_data = audio_data[:, 0]  # Take first channel if stereo

        # Validate audio data
        if len(audio_data) == 0:
            raise HTTPException(
                status_code=400, detail="Audio file appears to be empty or corrupted."
            )

        print(f"Analyzing audio: {len(audio_data)} samples, {sample_rate} Hz")
        if original_message:
            print(f"Original message provided: '{original_message}'")

        # Perform comprehensive analysis
        analysis_results = enhanced_audio_analysis(
            audio_data, sample_rate, file.filename
        )

        # Perform message decoding and comparison if original message is provided
        message_analysis = None
        if original_message:
            try:
                message_analysis = perform_message_comparison_analysis(
                    audio_data, sample_rate, original_message
                )
                analysis_results["message_analysis"] = message_analysis

                # Adjust steganography likelihood based on message comparison
                if message_analysis["messages_match"]:
                    # If messages match, increase confidence in steganography detection
                    analysis_results["steg_likelihood"] = min(
                        1.0, analysis_results["steg_likelihood"] + 0.3
                    )
                    analysis_results["detection_confidence"] = (
                        "Very High"
                        if analysis_results["steg_likelihood"] > 0.8
                        else "High"
                    )
                elif message_analysis["decoded_message"]:
                    # If a message was decoded but doesn't match, still indicates steganography
                    analysis_results["steg_likelihood"] = min(
                        1.0, analysis_results["steg_likelihood"] + 0.2
                    )
                    analysis_results["detection_confidence"] = (
                        "High"
                        if analysis_results["steg_likelihood"] > 0.7
                        else "Medium"
                    )

            except Exception as msg_error:
                print(f"Message comparison analysis failed: {msg_error}")
                analysis_results["message_analysis"] = {
                    "error": str(msg_error),
                    "messages_match": False,
                    "decoded_message": None,
                }

        # Use RL agent for additional analysis if available
        try:
            audio_features = rl_env.extract_audio_features(
                torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0)
            )
            rl_analysis = get_rl_steganography_assessment(audio_features)
            analysis_results["rl_assessment"] = rl_analysis
        except Exception as rl_error:
            print(f"RL analysis failed: {rl_error}")
            analysis_results["rl_assessment"] = None

        print(
            f"Analysis complete. Steganography likelihood: {analysis_results['steg_likelihood']:.3f}"
        )

        return JSONResponse(
            content={
                "analysis_results": analysis_results,
                "status": "success",
                "timestamp": datetime.now().isoformat(),
            }
        )

    except HTTPException:
        raise
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
            state = rl_env.reset()
            episode_reward = 0

            for step in range(10):  # steps per episode
                action = fallback_agent.choose_action(state)
                next_state, reward, done = rl_env.step(action)
                fallback_agent.learn(state, action, reward, next_state)
                episode_reward += reward
                state = next_state

                if done:
                    break

            print(f"Episode {ep + 1}: Reward = {episode_reward:.3f}")

        # Save the updated agent
        fallback_agent.save(legacy_agent_path)
        print(f"Agent training complete and saved to {legacy_agent_path}")

        return {
            "status": "training complete",
            "episodes": episodes,
            "agent_path": TRAINED_AGENT_PATH,
        }

    except Exception as e:
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
            q_value = fallback_agent.get_q(sample_state, method)
            scores[method] = q_value

        best_method = max(scores, key=scores.get) if scores else "LSB"

        return {
            "best_method": best_method,
            "method_scores": scores,
            "agent_loaded": os.path.exists(TRAINED_AGENT_PATH),
        }

    except Exception as e:
        return {
            "best_method": "LSB",
            "method_scores": {},
            "error": str(e),
            "agent_loaded": False,
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
        "agent_path": (
            TRAINED_AGENT_PATH
            if os.path.exists(TRAINED_AGENT_PATH)
            else FALLBACK_AGENT_PATH
        ),
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
        "agent_loaded": os.path.exists(TRAINED_AGENT_PATH)
        or os.path.exists(FALLBACK_AGENT_PATH),
    }


# ========== Entry Point ==========

if __name__ == "__main__":
    import uvicorn

    print("Starting Audio Steganography API with RL Agent Integration...")
    print(f"Trained agent available: {os.path.exists(TRAINED_AGENT_PATH)}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
