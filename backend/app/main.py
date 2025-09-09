from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
import librosa
import numpy as np
import io
import os


from typing import Annotated
from stable_baselines3 import PPO
from datetime import datetime

from app.core_modules.framework import RLAudioSteganography
from app.core_modules.preprocessor import AudioPreprocessor

app = FastAPI()

# Initialize RL agent
model_path = "app/ppo_audio_stego_model"
model = PPO.load(model_path)
framework = RLAudioSteganography()

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://audio-steganography-theta.vercel.app",
        "https://audio-steganography-git-main-kratosgados-projects.vercel.app/",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def read_root():
    return JSONResponse(
        content={
            "message": "Welcome to the audio steganography API!",
            "status": "success",
            "timestamp": datetime.now().isoformat(),
        }
    )


@app.get("/health")
async def health_check():
    """Health check endpoint for Docker and monitoring"""
    return JSONResponse(
        content={
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": "audio-steganography-api",
            "version": "1.0.0",
        }
    )


@app.post("/upload")
async def embed_message(
    file: Annotated[UploadFile, File()], message: Annotated[str, Form()] = "hello"
):
    """
    Embed a message in audio using the trained RL agent for optimal method selection.
    """
    try:
        print(f"message: {message}")
        waveform, sr = librosa.load(io.BytesIO(await file.read()))

        # AudioPreprocessor.save_audio(waveform, sr, "first.wav")

        framework.Initialize_components(method="spread-spectrum")
        # Read and preprocess the audio file

        audio_analysis = framework.get_audio_analysis(waveform)
        print(f"Audio analysis: {audio_analysis}")

        # Use spread spectrum embedding
        stego_audio = framework.embed_message(waveform, sr, message, model)
        print("Spread spectrum embedding successful")

        # Save the processed audio as WAV to preserve LSBs
        output_file = AudioPreprocessor.save_audio(stego_audio, sr)

        return StreamingResponse(
            output_file,
            media_type=file.content_type,
            # filename=output_file,
            headers={
                "Content-Disposition": "attachment; filename=output_file.wav",
                "X-Encoding-Method": "spread-spectrum",
                "X-Audio-Capacity": str(audio_analysis["practical_capacity_chars"]),
                "X-Audio-Duration": str(audio_analysis["duration_seconds"]),
                "Access-Control-Allow-Origin": "*",
            },
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
        waveform, sr = librosa.load(io.BytesIO(await file.read()))
        framework.Initialize_components(method="spread-spectrum")

        # Ensure audio_data is float32 and properly shaped
        if len(waveform.shape) == 1:
            waveform = waveform.astype(np.float32)
        else:
            waveform = waveform[:, 0].astype(np.float32)  # Take first channel if stereo

        print(f"Decoding audio: {len(waveform)} samples, {sr} Hz")

        # Use spread spectrum extraction
        decoded_message = ""
        decoding_method = "spread-spectrum"

        try:
            # The message length will be extracted from LSB during spread spectrum extraction
            decoded_message = framework.extract_message(waveform, sr, msg_length=None)
            print(f"Spread spectrum extracted: '{decoded_message}'")

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
            },
            headers={
                "Access-Control-Allow-Origin": "*",
            },
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
    # try:
    #     # Validate file type
    #     if not file.content_type or not file.content_type.startswith("audio/"):
    #         raise HTTPException(
    #             status_code=400,
    #             detail="Invalid file type. Please upload an audio file.",
    #         )
    #
    #     # Read the audio file
    #     file_content = await file.read()
    #     audio_data, sample_rate = sf.read(io.BytesIO(file_content))
    #
    #     # Ensure audio_data is properly shaped
    #     if len(audio_data.shape) > 1:
    #         audio_data = audio_data[:, 0]  # Take first channel if stereo
    #
    #     # Validate audio data
    #     if len(audio_data) == 0:
    #         raise HTTPException(
    #             status_code=400, detail="Audio file appears to be empty or corrupted."
    #         )
    #
    #     print(f"Analyzing audio: {len(audio_data)} samples, {sample_rate} Hz")
    #     if original_message:
    #         print(f"Original message provided: '{original_message}'")
    #
    #     # Perform comprehensive analysis
    #     analysis_results = enhanced_audio_analysis(
    #         audio_data, sample_rate, file.filename
    #     )
    #
    #     # Perform message decoding and comparison if original message is provided
    #     message_analysis = None
    #     if original_message:
    #         try:
    #             message_analysis = perform_message_comparison_analysis(
    #                 audio_data, sample_rate, original_message
    #             )
    #             analysis_results["message_analysis"] = message_analysis
    #
    #             # Adjust steganography likelihood based on message comparison
    #             if message_analysis["messages_match"]:
    #                 # If messages match, increase confidence in steganography detection
    #                 analysis_results["steg_likelihood"] = min(
    #                     1.0, analysis_results["steg_likelihood"] + 0.3
    #                 )
    #                 analysis_results["detection_confidence"] = (
    #                     "Very High"
    #                     if analysis_results["steg_likelihood"] > 0.8
    #                     else "High"
    #                 )
    #             elif message_analysis["decoded_message"]:
    #                 # If a message was decoded but doesn't match, still indicates steganography
    #                 analysis_results["steg_likelihood"] = min(
    #                     1.0, analysis_results["steg_likelihood"] + 0.2
    #                 )
    #                 analysis_results["detection_confidence"] = (
    #                     "High"
    #                     if analysis_results["steg_likelihood"] > 0.7
    #                     else "Medium"
    #                 )
    #
    #         except Exception as msg_error:
    #             print(f"Message comparison analysis failed: {msg_error}")
    #             analysis_results["message_analysis"] = {
    #                 "error": str(msg_error),
    #                 "messages_match": False,
    #                 "decoded_message": None,
    #             }
    #
    #     # Use RL agent for additional analysis if available
    #     try:
    #         audio_features = rl_env.extract_audio_features(
    #             torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0)
    #         )
    #         rl_analysis = get_rl_steganography_assessment(audio_features)
    #         analysis_results["rl_assessment"] = rl_analysis
    #     except Exception as rl_error:
    #         print(f"RL analysis failed: {rl_error}")
    #         analysis_results["rl_assessment"] = None
    #
    #     print(
    #         f"Analysis complete. Steganography likelihood: {analysis_results['steg_likelihood']:.3f}"
    #     )
    #
    #     return JSONResponse(
    #         content={
    #             "analysis_results": analysis_results,
    #             "status": "success",
    #             "timestamp": datetime.now().isoformat(),
    #         }
    #     )
    #
    # except HTTPException:
    #     raise
    # except Exception as e:
    #     print(f"Error in analysis: {e}")
    #     import traceback
    #
    #     traceback.print_exc()
    #     raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    # Placeholder response for now
    return JSONResponse(
        content={
            "analysis_results": {
                "steg_likelihood": 0.5,
                "detection_confidence": "Medium",
                "analysis_summary": {
                    "duration": 10.0,
                    "sample_rate": 22050,
                    "total_samples": 220500,
                    "channels": 1,
                },
            },
            "status": "success",
            "timestamp": datetime.now().isoformat(),
        }
    )


# ========== Entry Point ==========

if __name__ == "__main__":
    import uvicorn

    print("Starting Audio Steganography API with RL Agent Integration...")
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
