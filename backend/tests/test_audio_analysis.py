#!/usr/bin/env python3
"""
Test script for audio analysis and optimal action calculation
"""

import numpy as np
import librosa
import soundfile as sf
from core_modules.audio_analyzer import AudioAnalyzer
from core_modules.framework import RLAudioSteganography
from core_modules.config import cfg

def test_audio_analysis():
    """Test the audio analysis and optimal action calculation"""
    
    # Load test audio
    sample_rate = cfg.SAMPLE_RATE
    filename = "input-1.wav"
    test_audio, _ = librosa.load(filename, sr=sample_rate)
    
    print(f"Test audio: {len(test_audio)} samples, {sample_rate} Hz")
    print(f"Duration: {len(test_audio) / sample_rate:.2f} seconds")
    
    # Initialize audio analyzer
    analyzer = AudioAnalyzer()
    
    # Test message
    test_message = "Hello, this is a test message for spread spectrum steganography!"
    
    print(f"\n=== Audio Analysis ===")
    
    # Get audio features
    features = analyzer.extract_audio_features(test_audio, sample_rate)
    print("Audio Features:")
    for key, value in features.items():
        print(f"  {key}: {value:.4f}")
    
    # Calculate optimal actions
    print(f"\n=== Optimal Action Calculation ===")
    carrier_freq_norm, chip_rate_norm, snr_norm = analyzer.calculate_optimal_actions(
        test_audio, sample_rate, len(test_message)
    )
    
    print(f"Normalized Actions:")
    print(f"  Carrier Frequency: {carrier_freq_norm:.3f}")
    print(f"  Chip Rate: {chip_rate_norm:.3f}")
    print(f"  SNR: {snr_norm:.3f}")
    
    # Get capacity estimates
    print(f"\n=== Capacity Analysis ===")
    capacity_analysis = analyzer.get_embedding_capacity_estimate(test_audio, sample_rate)
    print("Capacity Analysis:")
    for key, value in capacity_analysis.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # Test with framework
    print(f"\n=== Framework Integration Test ===")
    framework = RLAudioSteganography()
    framework.Initialize_components(test_audio, method="spread-spectrum")
    
    try:
        # Embed message with calculated optimal actions
        print("Embedding message with optimal actions...")
        stego_audio = framework.embed_message(test_audio, test_message)
        print(f"Stego audio created: {len(stego_audio)} samples")
        
        # Save stego audio
        sf.write("test_stego_optimal.wav", stego_audio, sample_rate)
        print("Stego audio saved as 'test_stego_optimal.wav'")
        
        # Extract message
        print("Extracting message...")
        extracted_message = framework.extract_message(stego_audio, msg_length=None)
        print(f"Extracted message: '{extracted_message}'")
        
        # Compare results
        print(f"\n=== Results ===")
        print(f"Original message: '{test_message}'")
        print(f"Extracted message: '{extracted_message}'")
        print(f"Messages match: {test_message == extracted_message}")
        
        if test_message == extracted_message:
            print("✅ SUCCESS: Audio analysis and optimal actions working correctly!")
        else:
            print("❌ FAILURE: Messages don't match")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

def test_different_audio_types():
    """Test with different types of audio to see how actions vary"""
    
    print(f"\n{'='*60}")
    print("TESTING DIFFERENT AUDIO TYPES")
    print(f"{'='*60}")
    
    analyzer = AudioAnalyzer()
    sample_rate = cfg.SAMPLE_RATE
    
    # Test with different audio files if available
    test_files = ["input-1.wav", "input.wav"]
    
    for filename in test_files:
        try:
            print(f"\n--- Testing {filename} ---")
            test_audio, _ = librosa.load(filename, sr=sample_rate)
            
            # Get features
            features = analyzer.extract_audio_features(test_audio, sample_rate)
            print(f"Spectral Centroid: {features['spectral_centroid']:.0f} Hz")
            print(f"RMS Energy: {features['rms_energy']:.4f}")
            print(f"Zero Crossing Rate: {features['zero_crossing_rate']:.4f}")
            
            # Get optimal actions
            carrier_freq_norm, chip_rate_norm, snr_norm = analyzer.calculate_optimal_actions(
                test_audio, sample_rate, message_length=50
            )
            
            print(f"Optimal Actions:")
            print(f"  Carrier Freq: {carrier_freq_norm:.3f}")
            print(f"  Chip Rate: {chip_rate_norm:.3f}")
            print(f"  SNR: {snr_norm:.3f}")
            
        except Exception as e:
            print(f"Error testing {filename}: {e}")

if __name__ == "__main__":
    test_audio_analysis()
    test_different_audio_types() 