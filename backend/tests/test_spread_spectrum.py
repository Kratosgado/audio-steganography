#!/usr/bin/env python3
"""
Test script for spread spectrum steganography implementation
"""

import numpy as np
import librosa
import soundfile as sf
from core_modules.spread_spectrum import SpreadSpectrum
from core_modules.framework import RLAudioSteganography
from core_modules.config import cfg

def test_spread_spectrum():
    """Test the spread spectrum embedding and extraction"""
    
    # Create a simple test audio signal
    sample_rate = cfg.SAMPLE_RATE
    filename = "input-1.wav"
    test_audio, _ = librosa.load(filename, sr=sample_rate)
    
    # Test message
    test_message = "Hello, this is a test message for spread spectrum steganography!"
    
    print(f"Test audio: {len(test_audio)} samples, {sample_rate} Hz")
    print(f"Test message: '{test_message}'")
    print(f"Message length: {len(test_message)} characters")
    
    # Initialize framework
    framework = RLAudioSteganography()
    framework.Initialize_components(test_audio, method="spread-spectrum")
    
    try:
        # Embed message
        print("\n=== Embedding Message ===")
        stego_audio = framework.embed_message(test_audio, test_message)
        print(f"Stego audio created: {len(stego_audio)} samples")
        
        # Save stego audio for inspection
        sf.write("test_stego.wav", stego_audio, sample_rate)
        print("Stego audio saved as 'test_stego.wav'")
        
        # Extract message
        print("\n=== Extracting Message ===")
        extracted_message = framework.extract_message(stego_audio, msg_length=None)
        print(f"Extracted message: '{extracted_message}'")
        
        # Compare results
        print("\n=== Results ===")
        print(f"Original message: '{test_message}'")
        print(f"Extracted message: '{extracted_message}'")
        print(f"Messages match: {test_message == extracted_message}")
        
        if test_message == extracted_message:
            print("✅ SUCCESS: Spread spectrum steganography working correctly!")
        else:
            print("❌ FAILURE: Messages don't match")
            print(f"Original length: {len(test_message)}")
            print(f"Extracted length: {len(extracted_message)}")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_spread_spectrum() 