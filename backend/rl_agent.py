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

class StegoEnv:
    """Simple steganography environment for basic RL testing"""
    def __init__(self, encoding_methods=None):
        self.encoding_methods = encoding_methods or ['low_freq', 'mid_freq', 'high_freq']
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
        # Simple reward function - can be improved
        base_reward = random.uniform(0.3, 0.8)
        
        # Add some method-specific bias
        if method == 'low_freq':
            base_reward += 0.1
        elif method == 'mid_freq':
            base_reward += 0.05
        
        return min(base_reward, 1.0)

class QLearningAgent:
    """Q-Learning agent for steganography optimization"""
    
    def __init__(self, actions, alpha=0.1, gamma=0.95, epsilon=0.1):
        self.q_table = {}
        self.actions = actions
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        
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

class DeepQLearningAgent:
    """Placeholder for future deep Q-learning implementation"""
    def __init__(self, state_size, action_size, lr=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        # TODO: Implement neural network for DQN
        print("Deep Q-Learning agent not implemented yet. Using tabular Q-learning.")
        
    def choose_action(self, state):
        return random.randint(0, self.action_size - 1)
        
    def learn(self, state, action, reward, next_state, done):
        pass  # TODO: Implement DQN learning