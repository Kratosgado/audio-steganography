import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import librosa
import scipy.io.wavfile as wavfile
from collections import deque
import random

# Parameters
SAMPLE_RATE = 44100  # Audio sample rate
EMBEDDING_STEP = 100  # Embed one bit every 100 samples
MESSAGE = "101010"  # Example binary message to hide
GAMMA = 0.99  # Discount factor for RL
EPSILON = 1.0  # Exploration rate for epsilon-greedy
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.001
BATCH_SIZE = 32
MEMORY_SIZE = 1000
EPISODES = 100


# Policy Network (Decides how to modify audio samples)
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Environment Network (Simulates steganalysis feedback)
class EnvironmentNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(EnvironmentNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))  # Output: detection probability
        return x


# RL Agent
class RLAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.policy_net = PolicyNetwork(state_size, 128, action_size).float()
        self.target_net = PolicyNetwork(state_size, 128, action_size).float()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.epsilon = EPSILON

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state)
        return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q = self.target_net(next_states).max(1)[0]
        target_q = rewards + (1 - dones) * GAMMA * next_q

        loss = nn.MSELoss()(current_q, target_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


# Audio Steganography Environment
class AudioStegEnvironment:
    def __init__(self, audio_path, message):
        self.audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        self.audio = self.audio / np.max(np.abs(self.audio))  # Normalize
        self.message = [int(b) for b in message]
        self.pos = 0  # Current position in message
        self.sample_idx = 0  # Current sample index
        self.env_net = EnvironmentNetwork(
            input_size=10, hidden_size=64, output_size=1
        ).float()
        self.state_size = 10  # Example: 10 samples around current position
        self.action_size = 2  # Actions: modify sample (0: no change, 1: change)

    def reset(self):
        self.pos = 0
        self.sample_idx = 0
        return self.get_state()

    def get_state(self):
        # State: 10 samples around current index
        start = max(0, self.sample_idx - 5)
        end = min(len(self.audio), self.sample_idx + 5)
        state = np.zeros(10)
        state[: end - start] = self.audio[start:end]
        return state

    def step(self, action):
        # Action: 0 (no change), 1 (modify sample to embed bit)
        done = False
        reward = 0

        if self.pos >= len(self.message):
            done = True
            return self.get_state(), reward, done

        if self.sample_idx >= len(self.audio):
            done = True
            return self.get_state(), reward, done

        if action == 1:  # Modify sample to embed bit
            target_bit = self.message[self.pos]
            # Simple LSB-like modification (real-world would be more sophisticated)
            self.audio[self.sample_idx] = self.modify_sample(
                self.audio[self.sample_idx], target_bit
            )

        # Simulate steganalysis with environment network
        state = torch.FloatTensor(self.get_state()).unsqueeze(0)
        detection_prob = self.env_net(state).item()
        reward = 1 - detection_prob  # Reward: high if undetectable

        # Add quality reward (simplified SNR-like metric)
        original_audio = librosa.load("input.wav", sr=SAMPLE_RATE, mono=True)[0]
        original_audio = original_audio / np.max(np.abs(original_audio))
        if self.sample_idx < len(original_audio):
            snr = 10 * np.log10(
                np.mean(original_audio**2)
                / np.mean((self.audio - original_audio) ** 2 + 1e-10)
            )
            reward += 0.1 * snr  # Weight SNR contribution

        self.sample_idx += EMBEDDING_STEP
        if action == 1:
            self.pos += 1

        next_state = self.get_state()
        return next_state, reward, done

    def modify_sample(self, sample, bit):
        # Simplified modification: adjust sample to encode bit
        return sample + (0.01 if bit == 1 else -0.01)


# Main Training Loop
def main():
    env = AudioStegEnvironment("input.wav", MESSAGE)
    agent = RLAgent(state_size=10, action_size=2)

    for episode in range(EPISODES):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.train()
            state = next_state
            total_reward += reward

        agent.update_target_network()
        print(
            f"Episode {episode + 1}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}"
        )

        # Save modified audio
        wavfile.write(f"stego_audio_episode_{episode + 1}.wav", SAMPLE_RATE, env.audio)


if __name__ == "__main__":
    main()
