# from typing import Union

# from fastapi import FastAPI

# app = FastAPI()


# @app.get("/")
# def read_root():
#     return {"Hello": "World"}


# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Union[str, None] = None):
#     return {"item_id": item_id, "q": q}
from fastapi import FastAPI, UploadFile, File, Form
import numpy as np
import io
import wave
import os
from scipy.io import wavfile

app = FastAPI()
UPLOAD_FOLDER = "uploaded_files"
STEGO_FOLDER = "stego_files"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STEGO_FOLDER, exist_ok=True)

def embed_echo(audio_data, message, sr, delay=0.02, decay=0.5):
    """Embeds a binary message into an audio signal using echo hiding."""
    binary_msg = ''.join(format(ord(i), '08b') for i in message)
    audio = np.frombuffer(audio_data, dtype=np.int16)
    
    for i, bit in enumerate(binary_msg):
        if bit == '1':
            echo = np.roll(audio, int(delay * sr)) * decay
            audio += echo
    
    return audio.tobytes()

def extract_message(audio_data, sr, delay=0.02):
    """Extracts the hidden binary message from an audio file."""
    audio = np.frombuffer(audio_data, dtype=np.int16)
    extracted_bits = []
    step = int(delay * sr)
    
    for i in range(0, len(audio), step * 8):
        segment = audio[i:i + step]
        if np.mean(segment) > np.mean(audio[:step]):
            extracted_bits.append('1')
        else:
            extracted_bits.append('0')
    
    byte_str = ''.join(extracted_bits)
    message = ''.join(chr(int(byte_str[i:i+8], 2)) for i in range(0, len(byte_str), 8))
    return message

@app.get("/")
def read_root():
    return {"message": "Welcome to the Audio Steganography API!"}

@app.post("/upload/")
async def upload_audio(file: UploadFile = File(...)):
    """Endpoint to upload an audio file."""
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    return {"filename": file.filename, "size": os.path.getsize(file_path), "path": file_path}

@app.post("/embed/")
async def embed_message(file: UploadFile = File(...), message: str = Form(...)):
    """Endpoint to embed a message into an audio file using echo hiding."""
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    sr, audio_data = wavfile.read(file_path)
    stego_audio = embed_echo(audio_data.tobytes(), message, sr)
    
    stego_path = os.path.join(STEGO_FOLDER, "stego_" + file.filename)
    with wave.open(stego_path, "wb") as stego_file:
        stego_file.setnchannels(1)
        stego_file.setsampwidth(2)
        stego_file.setframerate(sr)
        stego_file.writeframes(stego_audio)
    
    return {"message": "Message embedded successfully.", "stego_audio_path": stego_path}

@app.post("/extract/")
async def extract_hidden_message(file: UploadFile = File(...)):
    """Endpoint to extract a hidden message from an audio file."""
    file_path = os.path.join(STEGO_FOLDER, file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    sr, audio_data = wavfile.read(file_path)
    message = extract_message(audio_data.tobytes(), sr)
    
    return {"extracted_message": message}
