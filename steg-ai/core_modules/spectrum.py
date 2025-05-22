import numpy as np
import soundfile as sf
import tensorflow as tf
from scipy.signal import stft, istft


class SpreadSpectrumStego:
    def __init__(self, fs=44100, nperseg=1024):
        self.fs = fs
        self.nperseg = nperseg
        self.key = None  # Pseudo-random noise (PRN) sequence

    def _generate_prn(self, length, seed=42):
        np.random.seed(seed)
        return np.random.choice([-1, 1], size=length)

    def _preprocess_message(self, message):
        # Convert message to binary array with header
        binary = "".join(format(ord(c), "08b") for c in message)
        binary = [int(b) for b in binary]
        header = [int(b) for b in format(len(message), "016b")]
        return np.array(header + binary)

    def embed(self, audio_path, message, output_path, alpha=0.01):
        # Load audio
        audio, fs = sf.read(audio_path)
        if audio.ndim > 1:
            audio = audio[:, 0]

        # STFT transformation
        f, t, Zxx = stft(audio, fs=self.fs, nperseg=self.nperseg)

        # Generate PRN sequence
        prn_length = len(Zxx.flatten())
        self.key = self._generate_prn(prn_length)

        # Prepare message with header
        msg_bits = self._preprocess_message(message)
        msg_signal = np.repeat(msg_bits, prn_length // len(msg_bits))[:prn_length]

        # Spread and modulate
        spread_msg = alpha * msg_signal * self.key
        Zxx_stego = Zxx.flatten() + spread_msg.reshape(Zxx.shape)

        # Inverse STFT
        _, audio_stego = istft(Zxx_stego, fs=self.fs, nperseg=self.nperseg)

        # Save stego audio
        sf.write(output_path, audio_stego, fs)

    def extract(self, stego_path):
        # Load stego audio
        audio_stego, fs = sf.read(stego_path)
        if audio_stego.ndim > 1:
            audio_stego = audio_stego[:, 0]

        # STFT transformation
        f, t, Zxx_stego = stft(audio_stego, fs=self.fs, nperseg=self.nperseg)

        # Correlate with PRN sequence
        correlation = (Zxx_stego.flatten() * self.key).reshape(-1, 16)
        decoded_bits = np.mean(correlation, axis=1) > 0

        # Extract header and message
        msg_length = int("".join(map(str, decoded_bits[:16].astype(int))), 2)
        msg_bits = decoded_bits[16 : 16 + msg_length * 8]
        chars = [
            chr(int("".join(map(str, msg_bits[i : i + 8])), 2))
            for i in range(0, len(msg_bits), 8)
        ]
        return "".join(chars)


class AIOptimizer:
    def __init__(self):
        self.model = self._build_model()

    def _build_model(self):
        # Neural network to predict optimal alpha (embedding strength)
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(100,)),  # Audio features
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid"),  # Output: alpha (0-0.1)
            ]
        )
        model.compile(optimizer="adam", loss="mse")
        return model

    def train(self, X, y):
        # X: Audio features (e.g., spectral flatness, RMS energy)
        # y: Optimal alpha values (precomputed from SNR/robustness tradeoff)
        self.model.fit(X, y, epochs=50, validation_split=0.2)

    def predict_alpha(self, features):
        return self.model.predict(features.reshape(1, -1))[0][0] * 0.1


# Example Usage
if __name__ == "__main__":
    # Initialize components
    stego = SpreadSpectrumStego()
    ai_optimizer = AIOptimizer()

    # Train AI model with synthetic data (replace with real measurements)
    X_train = np.random.randn(1000, 100)  # Simulated audio features
    y_train = np.random.uniform(0, 0.1, 1000)  # Optimal alphas
    ai_optimizer.train(X_train, y_train)

    # Embed message with AI-optimized alpha
    audio_features = np.random.randn(100)  # Replace with real feature extraction
    optimal_alpha = ai_optimizer.predict_alpha(audio_features)
    stego.embed("input.wav", "Secret123", "output.wav", alpha=optimal_alpha)

    # Extract message
    extracted = stego.extract("output.wav")
    print(f"Extracted message: {extracted}")
