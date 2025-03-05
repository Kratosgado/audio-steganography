from core_modules.lsb import AudioSteganography

if __name__ == "__main__":
    steg = AudioSteganography()
    original_audio = steg.load_audio("input_audio.flac")
    stego_audio = steg.lsb_embed(original_audio, "Hello, World!")

    steg.save_audio(stego_audio, "output_audio.wav")
    extracted_message = steg.lsb_extract(stego_audio, len("Hello, World!"))
    print(extracted_message)
