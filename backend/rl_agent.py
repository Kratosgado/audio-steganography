# import numpy as np
# import pickle
# import random

# class StegoEnv:
#     def __init__(self, encoding_methods):
#         self.encoding_methods = encoding_methods
#         self.current_state = None

#     def reset(self):
#         self.current_state = random.choice(self.encoding_methods)
#         return self.current_state

#     def step(self, action):
#         """Simulates using an encoding method (action) and evaluates it."""
#         reward = self.evaluate_encoding_method(action)
#         next_state = random.choice(self.encoding_methods)
#         done = False  # Optional: set to True if training ends
#         return next_state, reward, done

#     def evaluate_encoding_method(self, method):
#         """Mock performance metric: Replace with actual metrics from steganalysis."""
#         return random.uniform(0, 1)  # Replace with analysis quality metrics

# class QLearningAgent:
#     def __init__(self, actions, alpha=0.1, gamma=0.95, epsilon=0.1):
#         self.q_table = {}
#         self.actions = actions
#         self.alpha = alpha
#         self.gamma = gamma
#         self.epsilon = epsilon

#     def get_q(self, state, action):
#         return self.q_table.get((state, action), 0.0)

#     def choose_action(self, state):
#         if np.random.rand() < self.epsilon:
#             return random.choice(self.actions)
#         q_vals = [self.get_q(state, a) for a in self.actions]
#         return self.actions[int(np.argmax(q_vals))]

#     def learn(self, state, action, reward, next_state):
#         q_predict = self.get_q(state, action)
#         q_target = reward + self.gamma * max([self.get_q(next_state, a) for a in self.actions])
#         self.q_table[(state, action)] = q_predict + self.alpha * (q_target - q_predict)

#     def save(self, filepath):
#         with open(filepath, "wb") as f:
#             pickle.dump(self.q_table, f)

#     def load(self, filepath):
#         with open(filepath, "rb") as f:
#             self.q_table = pickle.load(f)

import numpy as np
import pickle
import random
import os
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
from typing import Dict, List, Tuple, Optional

# Experience tuple for DQN
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class AdvancedStegoEnv:
    """Advanced steganography environment with comprehensive evaluation"""
    def __init__(self, encoding_methods=None, audio_features_dim=50):
        self.encoding_methods = encoding_methods or [
            'neural', 'lsb', 'spread_spectrum', 'echo_hiding', 'phase_coding'
        ]
        self.audio_features_dim = audio_features_dim
        self.current_state = None
        self.current_audio = None
        self.current_message = None
        self.episode_step = 0
        self.max_episode_steps = 100
        
        # Parameter ranges for RL optimization
        self.param_ranges = {
            'embedding_strength': (0.001, 0.1),
            'bits_per_sample': (1, 4),
            'spreading_factor': (4, 16),
            'echo_delay_0': (20, 100),
            'echo_delay_1': (50, 200),
            'echo_decay': (0.1, 0.8),
            'segment_length': (512, 2048)
        }
        
    def reset(self, audio=None, message=None):
        """Reset environment with new audio and message"""
        self.current_audio = audio
        self.current_message = message
        self.episode_step = 0
        
        # Generate initial state (audio features + method encoding)
        if audio is not None:
            audio_features = self._extract_audio_features(audio)
        else:
            audio_features = np.random.randn(self.audio_features_dim - len(self.encoding_methods))
            
        # One-hot encode current method (start with random)
        method_encoding = np.zeros(len(self.encoding_methods))
        method_idx = random.randint(0, len(self.encoding_methods) - 1)
        method_encoding[method_idx] = 1.0
        
        self.current_state = np.concatenate([audio_features, method_encoding])
        return self.current_state
    
    def step(self, action_params):
        """Execute action with given parameters"""
        self.episode_step += 1
        
        # Decode action parameters
        method_idx = action_params.get('method_idx', 0)
        rl_params = self._decode_action_params(action_params)
        
        # Evaluate the steganographic embedding
        reward = self._evaluate_embedding(method_idx, rl_params)
        
        # Generate next state
        next_state = self._generate_next_state(method_idx)
        
        # Check if episode is done
        done = self.episode_step >= self.max_episode_steps
        
        self.current_state = next_state
        return next_state, reward, done, {'method': self.encoding_methods[method_idx]}
    
    def _extract_audio_features(self, audio):
        """Extract comprehensive audio features for state representation"""
        if isinstance(audio, torch.Tensor):
            audio_np = audio.squeeze().detach().cpu().numpy()
        else:
            audio_np = np.array(audio).flatten()
            
        features = []
        
        # Basic statistics
        features.extend([
            np.mean(audio_np),
            np.std(audio_np),
            np.max(audio_np),
            np.min(audio_np),
            np.median(audio_np)
        ])
        
        # Spectral features
        fft = np.fft.fft(audio_np)
        magnitude = np.abs(fft[:len(fft)//2])
        
        # Spectral centroid
        freqs = np.fft.fftfreq(len(audio_np))[:len(fft)//2]
        spectral_centroid = np.sum(freqs * magnitude) / (np.sum(magnitude) + 1e-10)
        features.append(spectral_centroid)
        
        # Spectral rolloff
        cumsum = np.cumsum(magnitude)
        rolloff_idx = np.where(cumsum >= 0.85 * cumsum[-1])[0]
        spectral_rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0
        features.append(spectral_rolloff)
        
        # Zero crossing rate
        zcr = np.sum(np.diff(np.sign(audio_np)) != 0) / len(audio_np)
        features.append(zcr)
        
        # Energy in different frequency bands
        n_bands = 10
        band_size = len(magnitude) // n_bands
        for i in range(n_bands):
            start_idx = i * band_size
            end_idx = min((i + 1) * band_size, len(magnitude))
            band_energy = np.sum(magnitude[start_idx:end_idx])
            features.append(band_energy)
        
        # Pad or truncate to desired length
        target_len = self.audio_features_dim - len(self.encoding_methods)
        if len(features) < target_len:
            features.extend([0.0] * (target_len - len(features)))
        else:
            features = features[:target_len]
            
        return np.array(features, dtype=np.float32)
    
    def _decode_action_params(self, action_params):
        """Decode action parameters to RL parameters"""
        rl_params = {}
        
        for param_name, (min_val, max_val) in self.param_ranges.items():
            if param_name in action_params:
                # Normalize from [0, 1] to parameter range
                normalized_val = action_params[param_name]
                rl_params[param_name] = min_val + normalized_val * (max_val - min_val)
                
        return rl_params
    
    def _evaluate_embedding(self, method_idx, rl_params):
        """Comprehensive evaluation of steganographic embedding"""
        if self.current_audio is None or self.current_message is None:
            return random.uniform(0.3, 0.7)  # Fallback for testing
            
        # Simulate embedding quality metrics
        base_reward = 0.0
        
        # Method-specific base rewards
        method_rewards = {
            0: 0.8,  # neural
            1: 0.6,  # lsb
            2: 0.7,  # spread_spectrum
            3: 0.65, # echo_hiding
            4: 0.75  # phase_coding
        }
        base_reward = method_rewards.get(method_idx, 0.5)
        
        # Parameter optimization rewards
        param_reward = 0.0
        
        # Embedding strength optimization
        if 'embedding_strength' in rl_params:
            strength = rl_params['embedding_strength']
            # Reward moderate embedding strengths
            if 0.005 <= strength <= 0.05:
                param_reward += 0.1
            elif strength < 0.001 or strength > 0.1:
                param_reward -= 0.1
                
        # Method-specific parameter rewards
        if method_idx == 1:  # LSB
            if 'bits_per_sample' in rl_params:
                bits = rl_params['bits_per_sample']
                if bits <= 2:  # Prefer fewer bits for better imperceptibility
                    param_reward += 0.05
                    
        elif method_idx == 2:  # Spread spectrum
            if 'spreading_factor' in rl_params:
                factor = rl_params['spreading_factor']
                if 6 <= factor <= 12:  # Optimal range
                    param_reward += 0.05
                    
        # Add noise for exploration
        noise = np.random.normal(0, 0.05)
        
        total_reward = base_reward + param_reward + noise
        return np.clip(total_reward, 0.0, 1.0)
    
    def _generate_next_state(self, method_idx):
        """Generate next state based on current action"""
        # Keep audio features, update method encoding
        audio_features = self.current_state[:-len(self.encoding_methods)]
        method_encoding = np.zeros(len(self.encoding_methods))
        method_encoding[method_idx] = 1.0
        
        return np.concatenate([audio_features, method_encoding])

class StegoEnv(AdvancedStegoEnv):
    """Backward compatibility wrapper"""
    def __init__(self, encoding_methods=None):
        super().__init__(encoding_methods)
        
    def evaluate_encoding_method(self, method):
        """Legacy method for backward compatibility"""
        method_idx = self.encoding_methods.index(method) if method in self.encoding_methods else 0
        return self._evaluate_embedding(method_idx, {})

class QLearningAgent:
    """Q-Learning agent for steganography optimization"""
    
    def __init__(self, actions, alpha=0.3, gamma=0.95, epsilon=0.3):
        self.q_table = {}
        self.actions = actions
        self.alpha = alpha  # Learning rate - increased for faster learning
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate - increased for more exploration
        
        # Track learning statistics
        self.total_episodes = 0
        self.total_steps = 0

    def get_q(self, state, action):
        """Get Q-value for state-action pair"""
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.rand() < self.epsilon:
            # Exploration: choose random action
            return random.choice(self.actions)
        else:
            # Exploitation: choose best action
            q_vals = [self.get_q(state, a) for a in self.actions]
            max_q = max(q_vals)
            
            # Handle ties by randomly selecting among best actions
            best_actions = [a for a, q in zip(self.actions, q_vals) if q == max_q]
            return random.choice(best_actions)

    def learn(self, state, action, reward, next_state):
        """Update Q-table using Q-learning update rule"""
        # Current Q-value
        q_predict = self.get_q(state, action)
        
        # Best Q-value for next state
        if next_state is not None:
            q_target = reward + self.gamma * max([self.get_q(next_state, a) for a in self.actions])
        else:
            # Terminal state
            q_target = reward
        
        # Q-learning update
        self.q_table[(state, action)] = q_predict + self.alpha * (q_target - q_predict)
        
        self.total_steps += 1

    def update_epsilon(self, episode, total_episodes):
        """Decay epsilon over time for less exploration as training progresses"""
        # Linear decay
        min_epsilon = 0.01
        decay_rate = (self.epsilon - min_epsilon) / (total_episodes * 0.8)
        self.epsilon = max(min_epsilon, self.epsilon - decay_rate)

    def get_policy(self):
        """Get current policy (best action for each state)"""
        policy = {}
        states = set([state for state, action in self.q_table.keys()])
        
        for state in states:
            q_vals = [self.get_q(state, a) for a in self.actions]
            best_action = self.actions[np.argmax(q_vals)]
            policy[state] = best_action
            
        return policy

    def get_statistics(self):
        """Get learning statistics"""
        return {
            'total_episodes': self.total_episodes,
            'total_steps': self.total_steps,
            'q_table_size': len(self.q_table),
            'epsilon': self.epsilon,
            'unique_states': len(set([state for state, action in self.q_table.keys()]))
        }

    def save(self, filepath):
        """Save the Q-table and agent parameters"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
            
            agent_data = {
                'q_table': self.q_table,
                'actions': self.actions,
                'alpha': self.alpha,
                'gamma': self.gamma,
                'epsilon': self.epsilon,
                'total_episodes': self.total_episodes,
                'total_steps': self.total_steps
            }
            
            with open(filepath, "wb") as f:
                pickle.dump(agent_data, f)
            
            print(f"Agent saved to {filepath}")
            return True
            
        except Exception as e:
            print(f"Error saving agent: {e}")
            return False

    def load(self, filepath):
        """Load the Q-table and agent parameters"""
        try:
            with open(filepath, "rb") as f:
                agent_data = pickle.load(f)
            
            self.q_table = agent_data.get('q_table', {})
            self.actions = agent_data.get('actions', self.actions)
            self.alpha = agent_data.get('alpha', self.alpha)
            self.gamma = agent_data.get('gamma', self.gamma)
            self.epsilon = agent_data.get('epsilon', self.epsilon)
            self.total_episodes = agent_data.get('total_episodes', 0)
            self.total_steps = agent_data.get('total_steps', 0)
            
            print(f"Agent loaded from {filepath}")
            print(f"Q-table size: {len(self.q_table)}")
            return True
            
        except Exception as e:
            print(f"Error loading agent: {e}")
            return False

    def print_q_table(self, max_entries=20):
        """Print Q-table for debugging"""
        print("\nQ-Table (showing top entries):")
        print("-" * 50)
        
        if not self.q_table:
            print("Q-table is empty")
            return
        
        # Sort by Q-value
        sorted_entries = sorted(self.q_table.items(), key=lambda x: x[1], reverse=True)
        
        for i, ((state, action), q_value) in enumerate(sorted_entries[:max_entries]):
            print(f"State: {state}, Action: {action}, Q-value: {q_value:.4f}")
        
        if len(sorted_entries) > max_entries:
            print(f"... and {len(sorted_entries) - max_entries} more entries")

class DQNNetwork(nn.Module):
    """Deep Q-Network for advanced RL-based steganography"""
    def __init__(self, state_size, action_size, hidden_sizes=[256, 128, 64]):
        super(DQNNetwork, self).__init__()
        
        layers = []
        input_size = state_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            input_size = hidden_size
            
        layers.append(nn.Linear(input_size, action_size))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class DeepQLearningAgent:
    """Deep Q-Learning agent for advanced steganography parameter optimization"""
    def __init__(self, state_size=55, action_size=7, learning_rate=0.001, 
                 discount_factor=0.95, epsilon=1.0, epsilon_decay=0.995, 
                 epsilon_min=0.01, memory_size=10000, batch_size=32):
        
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        
        # Experience replay memory
        self.memory = deque(maxlen=memory_size)
        
        # Neural networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network = DQNNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Copy weights to target network
        self.update_target_network()
        
        # Training parameters
        self.update_target_every = 100
        self.training_step = 0
        
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append(Experience(state, action, reward, next_state, done))
    
    def choose_action(self, state, training=True):
        """Choose action using epsilon-greedy policy with continuous action space"""
        if training and np.random.random() <= self.epsilon:
            # Random action - return normalized parameters [0, 1]
            return {
                'method_idx': np.random.randint(0, 5),  # 5 methods
                'embedding_strength': np.random.random(),
                'bits_per_sample': np.random.random(),
                'spreading_factor': np.random.random(),
                'echo_delay_0': np.random.random(),
                'echo_delay_1': np.random.random(),
                'echo_decay': np.random.random(),
                'segment_length': np.random.random()
            }
        else:
            # Use neural network to predict optimal parameters
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            
            # Convert Q-values to action parameters
            q_values_np = q_values.cpu().data.numpy()[0]
            
            return {
                'method_idx': int(np.argmax(q_values_np[:5])),  # First 5 for method selection
                'embedding_strength': torch.sigmoid(torch.tensor(q_values_np[5])).item(),
                'bits_per_sample': torch.sigmoid(torch.tensor(q_values_np[6])).item() if len(q_values_np) > 6 else np.random.random(),
                'spreading_factor': torch.sigmoid(torch.tensor(q_values_np[7])).item() if len(q_values_np) > 7 else np.random.random(),
                'echo_delay_0': torch.sigmoid(torch.tensor(q_values_np[8])).item() if len(q_values_np) > 8 else np.random.random(),
                'echo_delay_1': torch.sigmoid(torch.tensor(q_values_np[9])).item() if len(q_values_np) > 9 else np.random.random(),
                'echo_decay': torch.sigmoid(torch.tensor(q_values_np[10])).item() if len(q_values_np) > 10 else np.random.random(),
                'segment_length': torch.sigmoid(torch.tensor(q_values_np[11])).item() if len(q_values_np) > 11 else np.random.random()
            }
    
    def learn(self):
        """Train the neural network on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
            
        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e.state for e in batch]).to(self.device)
        actions = torch.LongTensor([e.action for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in batch]).to(self.device)
        dones = torch.BoolTensor([e.done for e in batch]).to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q-values from target network
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.discount_factor * next_q_values * ~dones)
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.training_step += 1
        if self.training_step % self.update_target_every == 0:
            self.update_target_network()
            
        # Decay epsilon
        self.update_epsilon()
        
        return loss.item()
    
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def update_epsilon(self):
        """Decay epsilon for exploration"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_model(self, filepath):
        """Save model to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'state_size': self.state_size,
            'action_size': self.action_size
        }, filepath)
    
    def load_model(self, filepath):
        """Load model from file"""
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.training_step = checkpoint['training_step']
            return True
        except FileNotFoundError:
            print(f"Model file {filepath} not found. Starting with fresh network.")
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

class RLSteganographyManager:
    """Manager class for RL-enhanced steganography"""
    def __init__(self, use_deep_rl=True):
        self.use_deep_rl = use_deep_rl
        self.env = AdvancedStegoEnv()
        
        if use_deep_rl:
            self.agent = DeepQLearningAgent(
                state_size=self.env.audio_features_dim,
                action_size=12  # Method selection + parameter optimization
            )
        else:
            # Define discrete actions for tabular Q-learning (method indices)
            actions = list(range(len(self.env.encoding_methods)))
            self.agent = QLearningAgent(actions=actions)
            
        self.training_history = []
        
    def optimize_embedding(self, audio, message, num_episodes=100):
        """Optimize embedding parameters using RL"""
        best_reward = -float('inf')
        best_params = None
        
        for episode in range(num_episodes):
            state = self.env.reset(audio, message)
            total_reward = 0
            
            for step in range(self.env.max_episode_steps):
                # Choose action
                if self.use_deep_rl:
                    action_params = self.agent.choose_action(state)
                    # Convert to discrete action for experience replay
                    action_idx = action_params['method_idx']
                else:
                    discrete_state = self._discretize_state(state)
                    action_idx = self.agent.choose_action(discrete_state)
                    action_params = {'method_idx': action_idx}
                
                # Take step
                next_state, reward, done, info = self.env.step(action_params)
                total_reward += reward
                
                # Store experience and learn
                if self.use_deep_rl:
                    self.agent.remember(state, action_idx, reward, next_state, done)
                    if len(self.agent.memory) > self.agent.batch_size:
                        self.agent.learn()
                else:
                    discrete_next_state = self._discretize_state(next_state)
                    self.agent.learn(
                        discrete_state, 
                        action_idx, 
                        reward, 
                        discrete_next_state
                    )
                    self.agent.update_epsilon()
                
                if reward > best_reward:
                    best_reward = reward
                    best_params = action_params.copy()
                
                state = next_state
                if done:
                    break
            
            self.training_history.append({
                'episode': episode,
                'total_reward': total_reward,
                'best_reward': best_reward,
                'epsilon': self.agent.epsilon
            })
            
            if episode % 10 == 0:
                print(f"Episode {episode}, Total Reward: {total_reward:.3f}, Best: {best_reward:.3f}")
        
        return best_params, best_reward
    
    def _discretize_state(self, state):
        """Convert continuous state to discrete for tabular Q-learning"""
        # Simple discretization - can be improved
        # Use a hash-based approach to create discrete states
        state_hash = hash(tuple(np.round(state, 2)))  # Round to 2 decimals for discretization
        return state_hash % 10000  # Limit to reasonable state space size
    
    def save_agent(self, filepath):
        """Save the trained agent"""
        self.agent.save_model(filepath)
    
    def load_agent(self, filepath):
        """Load a trained agent"""
        return self.agent.load_model(filepath)