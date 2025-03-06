from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware  # Correct import
from fastapi.responses import JSONResponse
import numpy as np
import soundfile as sf
import io
import os

app = FastAPI()

# CORS settings for the application
app.add_middleware(  
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (replace with specific domains in production)
    allow_credentials=True,  # Allow credentials (cookies, authorization headers)
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

def spread_spectrum_embed(audio_data, message, seed=42):
    np.random.seed(seed)
    message_length = len(message)
    audio_length = len(audio_data)
    
    # Generate a pseudo-random sequence
    pseudo_random_sequence = np.random.choice([-1, 1], size=audio_length)
    
    # Convert message to binary
    binary_message = ''.join(format(ord(char), '08b') for char in message)
    binary_message = np.array([int(bit) for bit in binary_message])
    
    # Repeat the message to match the audio length
    repeated_message = np.tile(binary_message, audio_length // message_length + 1)[:audio_length]
    
    # Embed the message using spread spectrum
    embedded_audio = audio_data + 0.01 * pseudo_random_sequence * repeated_message
    
    return embedded_audio

@app.post("/upload")
async def embed_message(file: UploadFile = File(...), message: str = "Hello"):
    # Read the uploaded .flac file
    audio_data, samplerate = sf.read(io.BytesIO(await file.read()))
    
    # Embed the message using spread spectrum
    embedded_audio = spread_spectrum_embed(audio_data, message)
    
    # Save the processed audio to a temporary file
    output_file = "output.flac"
    sf.write(output_file, embedded_audio, samplerate)
    
    # Return the processed audio file
    return FileResponse(output_file, media_type="audio/flac", filename=output_file)

def spread_spectrum_decode(audio_data, seed=42):
    np.random.seed(seed)
    audio_length = len(audio_data)
    
    # Generate the same pseudo-random sequence used during encoding
    pseudo_random_sequence = np.random.choice([-1, 1], size=audio_length)
    
    # Decode the message
    decoded_bits = (audio_data * pseudo_random_sequence) > 0
    decoded_bits = decoded_bits.astype(int)
    
    # Convert binary to string
    decoded_message = ''
    for i in range(0, len(decoded_bits), 8):
        byte_bits = decoded_bits[i:i+8]
        if len(byte_bits) == 8:
            byte = int(''.join(map(str, byte_bits)), 2)
            decoded_message += chr(byte)
    
    return decoded_message

    
@app.post("/decode")
async def decode_message(file: UploadFile = File(...)):
    # Read the uploaded .flac file
    audio_data, samplerate = sf.read(io.BytesIO(await file.read()))
    
    # Decode the message using spread spectrum
    decoded_message = spread_spectrum_decode(audio_data)
    
    # Return the decoded message
    return JSONResponse(content={"decoded_message": decoded_message})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)