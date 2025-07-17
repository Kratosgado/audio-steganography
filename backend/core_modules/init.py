import librosa
from config import cfg
from framework import RLAudioSteganography
framework = RLAudioSteganography(cfg)

# --- Training Phase ---

# Initialize components with the training audio
framework.Initialize_components(simple_audio_files[0], method="spread-spectrum")

# Train PPO agent
print("Training PPO agent...")
model = framework.train_ppo()

# Save the trained model
model_save_path = "ppo_audio_stego_model"
model.save(model_save_path)
print(f"Trained model saved to {model_save_path}")

# --- Embedding with the Trained Model on a New Audio ---
print("\nEmbedding message in a new audio file using the trained model...")


# Define a new audio file and output path
new_audio_path = librosa.ex('trumpet', hq=True) # Using a different trumpet example
new_output_path = "stego_new_audio.wav"

# Initialize components with the *new* audio
framework.Initialize_components(new_audio_path, method="spread-spectrum")

message_to_embed = "Just a try and error"

# Embed message using the loaded model
framework.embed_message(new_audio_path, message_to_embed, new_output_path, loaded_model)

# Extract message from the new stego audio
extracted_message = framework.extract_message(new_output_path, len(message_to_embed))
print(f"\nOriginal Message: {message_to_embed}")
print(f"Extracted Message from new audio: {extracted_message}")