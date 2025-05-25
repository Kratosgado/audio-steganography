import numpy as np
import pickle
import random

class StegoEnv:
    def __init__(self, encoding_methods):
        self.encoding_methods = encoding_methods
        self.current_state = None

    def reset(self):
        self.current_state = random.choice(self.encoding_methods)
        return self.current_state

    def step(self, action):
        """Simulates using an encoding method (action) and evaluates it."""
        reward = self.evaluate_encoding_method(action)
        next_state = random.choice(self.encoding_methods)
        done = False  # Optional: set to True if training ends
        return next_state, reward, done

    def evaluate_encoding_method(self, method):
        """Mock performance metric: Replace with actual metrics from steganalysis."""
        return random.uniform(0, 1)  # Replace with analysis quality metrics

class QLearningAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.95, epsilon=0.1):
        self.q_table = {}
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_q(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.choice(self.actions)
        q_vals = [self.get_q(state, a) for a in self.actions]
        return self.actions[int(np.argmax(q_vals))]

    def learn(self, state, action, reward, next_state):
        q_predict = self.get_q(state, action)
        q_target = reward + self.gamma * max([self.get_q(next_state, a) for a in self.actions])
        self.q_table[(state, action)] = q_predict + self.alpha * (q_target - q_predict)

    def save(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self.q_table, f)

    def load(self, filepath):
        with open(filepath, "rb") as f:
            self.q_table = pickle.load(f)
