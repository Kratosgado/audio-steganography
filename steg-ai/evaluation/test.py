import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split


# Generate synthetic dataset (replace with real data)
def generate_data(num_samples=1000):
    X = np.random.randn(
        num_samples, 1000
    )  # Audio features (e.g., spectral coefficients)
    y_snr = np.random.uniform(10, 50, num_samples)  # Simulated SNR values
    y_mse = np.random.uniform(0, 0.1, num_samples)  # Simulated MSE values
    return X, y_snr, y_mse


# Build a model to predict SNR and MSE
model = tf.keras.Sequential(
        tf.keras.layers.Dense(64, activation="relu", input_shape=(1000,)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(2),  # Output: SNR and MSE
    ]
)

model.compile(optimizer="adam", loss="mse")

# Train-test split
X, y_snr, y_mse = generate_data()
y = np.vstack((y_snr, y_mse)).T
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the model
model.fit(X_train, y_train, epochs=10, validation_split=0.1)

# Predict SNR/MSE for a new sample
sample = np.random.randn(1, 1000)
predicted_snr, predicted_mse = model.predict(sample)
print(f"Predicted SNR: {predicted_snr[0]}, Predicted MSE: {predicted_mse[0]}")
