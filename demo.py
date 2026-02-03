"""
Continual Learning Demonstration using PyTorch and OpenAI Gym.

This script demonstrates a minimal continual learning system with:
1. Simple neural networks (Feed-forward or GRU)
2. OpenAI Gym environment (CartPole)
3. Recurrent feedback loop (outputs affect environment, which feeds back as inputs)
4. Real-time visualizations of learning dynamics
"""
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
import argparse

from networks import FeedForwardNetwork, GRUNetwork
from visualization import ContinualLearningVisualizer


class ContinualLearningAgent:
    """Agent that learns continually from environment interactions."""
    
    def __init__(self, network_type='feedforward', learning_rate=0.01):
        # Initialize environment
        self.env = gym.make('CartPole-v1')
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        
        # Initialize network
        hidden_size = 32
        if network_type == 'gru':
            self.network = GRUNetwork(self.state_size, hidden_size, self.action_size)
            self.hidden_state = None
        else:
            self.network = FeedForwardNetwork(self.state_size, hidden_size, self.action_size)
        
        self.network_type = network_type
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Initialize visualizer
        self.visualizer = ContinualLearningVisualizer()
        
        # Metrics
        self.episode_rewards = []
        self.total_steps = 0
    
    def select_action(self, state):
        """Select action using current network policy."""
        state_tensor = torch.FloatTensor(state)
        
        with torch.no_grad():
            if self.network_type == 'gru':
                action_probs, self.hidden_state = self.network(state_tensor, self.hidden_state)
                action_probs = action_probs.squeeze()  # Remove batch dimension
            else:
                action_probs = self.network(state_tensor)
        
        # Use softmax to get probabilities
        probabilities = torch.softmax(action_probs, dim=-1)
        
        # Sample action from probability distribution
        action = torch.multinomial(probabilities, 1).item()
        
        return action, probabilities.numpy()
    
    def update_network(self, state, action, reward, next_state, done):
        """
        Update network based on experience.
        Implements online learning with immediate feedback.
        """
        state_tensor = torch.FloatTensor(state)
        next_state_tensor = torch.FloatTensor(next_state)
        
        # Compute current Q-values
        if self.network_type == 'gru':
            current_q, _ = self.network(state_tensor, self.hidden_state)
            current_q = current_q.squeeze()  # Remove batch dimension
        else:
            current_q = self.network(state_tensor)
        
        # Compute target Q-values using simple TD-learning
        with torch.no_grad():
            if self.network_type == 'gru':
                next_q, _ = self.network(next_state_tensor, self.hidden_state)
                next_q = next_q.squeeze()  # Remove batch dimension
            else:
                next_q = self.network(next_state_tensor)
            
            target_q = current_q.clone()
            if done:
                target_q[action] = reward
            else:
                # Bellman equation: Q(s,a) = r + gamma * max(Q(s',a'))
                target_q[action] = reward + 0.99 * torch.max(next_q)
        
        # Compute loss and update
        loss = self.criterion(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def run_episode(self, episode_num, max_steps=500):
        """
        Run a single episode of continual learning.
        This implements the recurrent loop where:
        1. Network outputs action
        2. Action affects environment
        3. Environment provides new state
        4. New state feeds back as network input
        5. Network learns online from this experience
        """
        state, _ = self.env.reset()
        
        # Reset hidden state for GRU
        if self.network_type == 'gru':
            self.hidden_state = self.network.init_hidden()
        
        episode_reward = 0
        step = 0
        
        while step < max_steps:
            # Network output affects environment
            action, action_probs = self.select_action(state)
            
            # Environment responds to network output
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            
            # Online learning: update network immediately
            loss = self.update_network(state, action, reward, next_state, done)
            
            # Update visualizations
            self.visualizer.update(loss, episode_reward, action, action_probs)
            
            # Periodic visualization updates (every 10 steps)
            if self.total_steps % 10 == 0:
                self.visualizer.plot(episode_num)
            
            state = next_state
            step += 1
            self.total_steps += 1
            
            if done:
                break
        
        self.episode_rewards.append(episode_reward)
        return episode_reward, step
    
    def train(self, num_episodes=100):
        """Train the agent for multiple episodes."""
        print(f"\nStarting Continual Learning with {self.network_type.upper()} network")
        print(f"{'='*60}")
        
        for episode in range(num_episodes):
            reward, steps = self.run_episode(episode)
            
            # Print progress
            if episode % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else np.mean(self.episode_rewards)
                print(f"Episode {episode:3d} | Steps: {steps:3d} | "
                      f"Reward: {reward:6.1f} | Avg(10): {avg_reward:6.1f}")
        
        # Final visualization
        self.visualizer.plot(num_episodes)
        self.visualizer.save('continual_learning_results.png')
        
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Total steps: {self.total_steps}")
        print(f"Average reward: {np.mean(self.episode_rewards):.2f}")
        print(f"Final 10 episodes average: {np.mean(self.episode_rewards[-10:]):.2f}")
        
        return self.episode_rewards
    
    def close(self):
        """Clean up resources."""
        self.env.close()
        self.visualizer.close()


def main():
    """Main entry point for the continual learning demonstration."""
    parser = argparse.ArgumentParser(description='Continual Learning Demonstration')
    parser.add_argument('--network', type=str, default='feedforward',
                        choices=['feedforward', 'gru'],
                        help='Network architecture to use (default: feedforward)')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of episodes to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate (default: 0.01)')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("CONTINUAL LEARNING DEMONSTRATION")
    print("="*60)
    print(f"Network Type: {args.network.upper()}")
    print(f"Episodes: {args.episodes}")
    print(f"Learning Rate: {args.lr}")
    print("="*60)
    
    # Create and train agent
    agent = ContinualLearningAgent(network_type=args.network, learning_rate=args.lr)
    
    try:
        agent.train(num_episodes=args.episodes)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    finally:
        agent.close()
    
    print("\nDemonstration complete! Check 'continual_learning_results.png' for visualizations.")


if __name__ == '__main__':
    main()
