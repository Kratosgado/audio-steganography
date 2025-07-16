# import numpy as np
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# import torch
# from rl_environment import AudioStegEnvironment
# from rl_agent import QLearningAgent
# from utils import load_audio, text_to_bits
# import os

# class TrainingAnalyzer:
#     def __init__(self):
#         self.rewards_history = []
#         self.accuracy_history = []
#         self.snr_history = []
#         self.susceptibility_scores = []
#         self.episode_metrics = {}
    
#     def update_metrics(self, episode, reward, accuracy, snr, susceptibility):
#         if episode not in self.episode_metrics:
#             self.episode_metrics[episode] = {
#                 'rewards': [],
#                 'accuracies': [],
#                 'snrs': [],
#                 'susceptibilities': []
#             }
        
#         metrics = self.episode_metrics[episode]
#         metrics['rewards'].append(reward)
#         metrics['accuracies'].append(accuracy)
#         metrics['snrs'].append(snr)
#         metrics['susceptibilities'].append(susceptibility)
    
#     def calculate_episode_summary(self, episode):
#         metrics = self.episode_metrics[episode]
#         self.rewards_history.append(np.mean(metrics['rewards']))
#         self.accuracy_history.append(np.mean(metrics['accuracies']))
#         self.snr_history.append(np.mean(metrics['snrs']))
#         self.susceptibility_scores.append(np.mean(metrics['susceptibilities']))
    
#     def plot_metrics(self, save_dir='training_plots'):
#         os.makedirs(save_dir, exist_ok=True)
        
#         # Plot average reward per episode
#         plt.figure(figsize=(10, 6))
#         plt.plot(self.rewards_history)
#         plt.title('Average Reward per Episode')
#         plt.xlabel('Episode')
#         plt.ylabel('Reward')
#         plt.savefig(f'{save_dir}/rewards.png')
#         plt.close()
        
#         # Plot message recovery accuracy
#         plt.figure(figsize=(10, 6))
#         plt.plot(self.accuracy_history)
#         plt.title('Message Recovery Accuracy')
#         plt.xlabel('Episode')
#         plt.ylabel('Accuracy')
#         plt.savefig(f'{save_dir}/accuracy.png')
#         plt.close()
        
#         # Plot SNR
#         plt.figure(figsize=(10, 6))
#         plt.plot(self.snr_history)
#         plt.title('Signal-to-Noise Ratio')
#         plt.xlabel('Episode')
#         plt.ylabel('SNR (dB)')
#         plt.savefig(f'{save_dir}/snr.png')
#         plt.close()
        
#         # Plot susceptibility score
#         plt.figure(figsize=(10, 6))
#         plt.plot(self.susceptibility_scores)
#         plt.title('Steganalysis Susceptibility')
#         plt.xlabel('Episode')
#         plt.ylabel('Susceptibility Score')
#         plt.savefig(f'{save_dir}/susceptibility.png')
#         plt.close()

# def calculate_susceptibility(stego_audio, original_audio):
#     """Calculate how susceptible the steganography is to detection"""
#     # Frequency domain analysis
#     stego_fft = torch.fft.fft(stego_audio)
#     orig_fft = torch.fft.fft(original_audio)
    
#     # Calculate spectral differences
#     spectral_diff = torch.abs(stego_fft - orig_fft)
    
#     # Statistical analysis
#     mean_diff = torch.mean(spectral_diff)
#     std_diff = torch.std(spectral_diff)
    
#     # Combine metrics into a susceptibility score (0-1, lower is better)
#     susceptibility = torch.tanh(mean_diff + std_diff).item()
    
#     return susceptibility

# def train_agent(num_episodes=1000, audio_path='path_to_audio.wav', message='Hello World'):
#     # Initialize environment, agent and analyzer
#     env = AudioStegEnvironment(audio_path=audio_path, message=message)
#     actions = ['low_freq', 'mid_freq', 'high_freq']
#     agent = QLearningAgent(actions=actions)
#     analyzer = TrainingAnalyzer()
    
#     # Load original audio for comparison
#     original_audio = load_audio(audio_path)
    
#     # Training loop
#     for episode in tqdm(range(num_episodes)):
#         state = env.reset()
#         done = False
#         episode_reward = 0
        
#         while not done:
#             # Agent selects and performs action
#             action = agent.choose_action(state)
#             next_state, reward, done = env.step(action)
            
#             # Get stego audio and decoded message
#             stego_audio = env.get_current_audio()
#             decoded_message = env.decode_message()
            
#             # Calculate metrics
#             snr = env.calculate_snr(original_audio, stego_audio)
#             accuracy = env.calculate_bit_accuracy(text_to_bits(message), decoded_message)
#             susceptibility = calculate_susceptibility(stego_audio, original_audio)
            
#             # Update analyzer
#             analyzer.update_metrics(episode, reward, accuracy, snr, susceptibility)
            
#             # Agent learns from experience
#             agent.learn(state, action, reward, next_state)
#             state = next_state
#             episode_reward += reward
        
#         # Calculate episode summary
#         analyzer.calculate_episode_summary(episode)
        
#         # Save agent periodically
#         if (episode + 1) % 100 == 0:
#             agent.save(f'agent_checkpoint_{episode+1}.pkl')
    
#     # Plot final metrics
#     analyzer.plot_metrics()
    
#     # Save final agent
#     agent.save('trained_agent_final.pkl')
    
#     return analyzer

# if __name__ == '__main__':
#     # Example usage
#     audio_path = '174-84280-0001.flac'
#     message = 'This is a secret message'
#     analyzer = train_agent(num_episodes=1000, audio_path=audio_path, message=message)

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import os
import sys

# Try to import custom modules with error handling
try:
    from rl_environment import AudioStegEnvironment
    from rl_agent import QLearningAgent
    from utils import load_audio, text_to_bits, create_synthetic_audio
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all files are in the same directory")
    sys.exit(1)

class TrainingAnalyzer:
    def __init__(self):
        self.rewards_history = []
        self.accuracy_history = []
        self.snr_history = []
        self.susceptibility_scores = []
        self.episode_metrics = {}
    
    def update_metrics(self, episode, reward, accuracy, snr, susceptibility):
        if episode not in self.episode_metrics:
            self.episode_metrics[episode] = {
                'rewards': [],
                'accuracies': [],
                'snrs': [],
                'susceptibilities': []
            }
        
        metrics = self.episode_metrics[episode]
        metrics['rewards'].append(reward)
        metrics['accuracies'].append(accuracy)
        metrics['snrs'].append(snr)
        metrics['susceptibilities'].append(susceptibility)
    
    def calculate_episode_summary(self, episode):
        if episode in self.episode_metrics:
            metrics = self.episode_metrics[episode]
            self.rewards_history.append(np.mean(metrics['rewards']) if metrics['rewards'] else 0)
            self.accuracy_history.append(np.mean(metrics['accuracies']) if metrics['accuracies'] else 0)
            self.snr_history.append(np.mean(metrics['snrs']) if metrics['snrs'] else 0)
            self.susceptibility_scores.append(np.mean(metrics['susceptibilities']) if metrics['susceptibilities'] else 0)
        else:
            # Add zeros if no metrics recorded
            self.rewards_history.append(0)
            self.accuracy_history.append(0)
            self.snr_history.append(0)
            self.susceptibility_scores.append(0)
    
    def plot_metrics(self, save_dir='training_plots'):
        try:
            os.makedirs(save_dir, exist_ok=True)
            
            if not self.rewards_history:
                print("No metrics to plot")
                return
            
            # Plot average reward per episode
            plt.figure(figsize=(10, 6))
            plt.plot(self.rewards_history)
            plt.title('Average Reward per Episode')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.grid(True)
            plt.savefig(f'{save_dir}/rewards.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            # Plot message recovery accuracy
            plt.figure(figsize=(10, 6))
            plt.plot(self.accuracy_history)
            plt.title('Message Recovery Accuracy')
            plt.xlabel('Episode')
            plt.ylabel('Accuracy')
            plt.grid(True)
            plt.savefig(f'{save_dir}/accuracy.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            # Plot SNR
            plt.figure(figsize=(10, 6))
            plt.plot(self.snr_history)
            plt.title('Signal-to-Noise Ratio')
            plt.xlabel('Episode')
            plt.ylabel('SNR (dB)')
            plt.grid(True)
            plt.savefig(f'{save_dir}/snr.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            # Plot susceptibility score
            plt.figure(figsize=(10, 6))
            plt.plot(self.susceptibility_scores)
            plt.title('Steganalysis Susceptibility')
            plt.xlabel('Episode')
            plt.ylabel('Susceptibility Score')
            plt.grid(True)
            plt.savefig(f'{save_dir}/susceptibility.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Plots saved to {save_dir}/")
            
        except Exception as e:
            print(f"Error plotting metrics: {e}")

def calculate_susceptibility(stego_audio, original_audio):
    """Calculate how susceptible the steganography is to detection"""
    try:
        # Ensure tensors
        if not isinstance(stego_audio, torch.Tensor):
            stego_audio = torch.tensor(stego_audio)
        if not isinstance(original_audio, torch.Tensor):
            original_audio = torch.tensor(original_audio)
        
        # Simple statistical analysis
        orig_mean = torch.mean(original_audio)
        stego_mean = torch.mean(stego_audio)
        
        orig_std = torch.std(original_audio)
        stego_std = torch.std(stego_audio)
        
        # Calculate difference in statistics (normalized)
        mean_diff = abs(orig_mean - stego_mean) / (abs(orig_mean) + 1e-6)
        std_diff = abs(orig_std - stego_std) / (orig_std + 1e-6)
        
        susceptibility = (mean_diff + std_diff).item()
        return min(susceptibility, 1.0)
        
    except Exception as e:
        print(f"Error calculating susceptibility: {e}")
        return 0.5

def train_agent(num_episodes=100, audio_path=None, message='Hello World', max_steps_per_episode=10):
    """Train the RL agent for audio steganography"""
    print("Initializing training environment...")
    
    # Check if audio file exists, if not use synthetic audio
    if audio_path and not os.path.exists(audio_path):
        print(f"Audio file {audio_path} not found. Using synthetic audio.")
        audio_path = None
    
    # Initialize environment and agent
    try:
        env = AudioStegEnvironment(audio_path=audio_path, message=message, max_steps=max_steps_per_episode)
        actions = [0, 1, 2]  # Discrete actions for frequency bands
        agent = QLearningAgent(actions=actions, alpha=0.1, gamma=0.95, epsilon=0.3)
        analyzer = TrainingAnalyzer()
        
        print(f"Training for {num_episodes} episodes...")
        print(f"Message: '{message}'")
        print(f"Audio source: {'File' if audio_path else 'Synthetic'}")
        
    except Exception as e:
        print(f"Error initializing environment: {e}")
        return None
    
    # Get original audio for comparison
    try:
        if audio_path:
            original_audio = load_audio(audio_path)
        else:
            original_audio = create_synthetic_audio()
    except Exception as e:
        print(f"Error loading audio: {e}")
        original_audio = torch.randn(1, 16000)
    
    # Training loop
    best_reward = float('-inf')
    
    for episode in tqdm(range(num_episodes), desc="Training"):
        try:
            state = env.reset()
            done = False
            episode_reward = 0
            step_count = 0
            
            while not done and step_count < max_steps_per_episode:
                # Agent selects and performs action
                action = agent.choose_action(state)
                next_state, reward, done, _ = env.step(action)
                
                # Get current metrics
                try:
                    stego_audio = env.get_current_audio()
                    decoded_message = env.decode_message()
                    
                    # Calculate metrics
                    snr = env.calculate_snr(original_audio, stego_audio)
                    accuracy = env.calculate_bit_accuracy(text_to_bits(message), decoded_message)
                    susceptibility = calculate_susceptibility(stego_audio, original_audio)
                    
                    # Update analyzer
                    analyzer.update_metrics(episode, reward, accuracy, snr, susceptibility)
                    
                except Exception as e:
                    print(f"Error calculating metrics in episode {episode}: {e}")
                    # Use default values
                    analyzer.update_metrics(episode, reward, 0.0, 0.0, 0.5)
                
                # Agent learns from experience
                agent.learn(state, action, reward, next_state)
                state = next_state
                episode_reward += reward
                step_count += 1
            
            # Track best performance
            if episode_reward > best_reward:
                best_reward = episode_reward
            
            # Calculate episode summary
            analyzer.calculate_episode_summary(episode)
            
            # Print progress every 10 episodes
            if (episode + 1) % 10 == 0:
                recent_rewards = analyzer.rewards_history[-10:] if len(analyzer.rewards_history) >= 10 else analyzer.rewards_history
                avg_reward = np.mean(recent_rewards) if recent_rewards else 0
                print(f"Episode {episode + 1}: Avg Reward = {avg_reward:.3f}, Best = {best_reward:.3f}")
            
            # Save agent periodically
            if (episode + 1) % 50 == 0:
                try:
                    agent.save(f'checkpoints/agent_checkpoint_{episode+1}.pkl')
                except Exception as e:
                    print(f"Error saving checkpoint: {e}")
            
            # Decay epsilon for exploration
            if episode > num_episodes * 0.7:  # Start decaying after 70% of episodes
                agent.epsilon = max(0.01, agent.epsilon * 0.995)
                
        except Exception as e:
            print(f"Error in episode {episode}: {e}")
            # Add default metrics for failed episodes
            analyzer.rewards_history.append(-1.0)
            analyzer.accuracy_history.append(0.0)
            analyzer.snr_history.append(0.0)
            analyzer.susceptibility_scores.append(1.0)
            continue
    
    print("Training completed!")
    
    # Plot final metrics
    try:
        analyzer.plot_metrics()
    except Exception as e:
        print(f"Error plotting metrics: {e}")
    
    # Save final agent
    try:
        os.makedirs('models', exist_ok=True)
        agent.save('models/trained_agent_final.pkl')
        print("Final model saved to models/trained_agent_final.pkl")
    except Exception as e:
        print(f"Error saving final model: {e}")
    
    # Print final statistics
    if analyzer.rewards_history:
        print(f"\nTraining Statistics:")
        print(f"Final average reward: {np.mean(analyzer.rewards_history[-10:]):.3f}")
        print(f"Best reward: {best_reward:.3f}")
        print(f"Final accuracy: {analyzer.accuracy_history[-1]:.3f}")
        print(f"Final SNR: {analyzer.snr_history[-1]:.3f}")
    
    return analyzer

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('training_plots', exist_ok=True)
    
    # Configuration
    audio_file = '174-84280-0001.flac'  # Will use synthetic if not found
    secret_message = 'This is a secret message for steganography training'
    num_training_episodes = 200
    
    print("Starting Audio Steganography RL Training")
    print("=" * 50)
    
    # Train the agent
    try:
        analyzer = train_agent(
            num_episodes=num_training_episodes,
            audio_path=audio_file,
            message=secret_message,
            max_steps_per_episode=20
        )
        
        if analyzer:
            print("\nTraining completed successfully!")
        else:
            print("\nTraining failed!")
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()