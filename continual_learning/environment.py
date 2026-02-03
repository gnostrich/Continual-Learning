"""Simulated environment for continual learning."""

import torch
import numpy as np


class Environment:
    """
    Simulated environment that receives actions and returns observations.
    
    The environment:
    - Receives actions from the model
    - Updates internal state
    - Returns multi-modal observations (external signals)
    - Provides feedback signals (internal signals)
    """
    
    def __init__(
        self,
        observation_dim: int = 64,
        action_dim: int = 32,
        num_modalities: int = 2,
        task_difficulty: float = 0.5,
    ):
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.num_modalities = num_modalities
        self.task_difficulty = task_difficulty
        
        # Environment state
        self.state = np.random.randn(observation_dim)
        self.time_step = 0
        
        # Task parameters (changes over time for continual learning)
        self.current_task = 0
        self.task_params = np.random.randn(observation_dim)
        
    def reset(self):
        """Reset environment to initial state."""
        self.state = np.random.randn(self.observation_dim)
        self.time_step = 0
        return self.get_observation()
        
    def step(self, action):
        """
        Execute action and return observation.
        
        Args:
            action: Action tensor from model [batch, action_dim]
            
        Returns:
            observations: List of observation tensors (multi-modal)
            reward: Scalar reward signal
            done: Whether episode is complete
            info: Additional information dict
        """
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        
        # Update environment state based on action
        # Pad or truncate action to match observation dimension
        action_flat = action.mean(axis=0)
        if len(action_flat) < self.observation_dim:
            # Pad with zeros
            action_effect = np.zeros(self.observation_dim)
            action_effect[:len(action_flat)] = action_flat
        else:
            # Truncate
            action_effect = action_flat[:self.observation_dim]
        self.state = 0.9 * self.state + 0.1 * action_effect
        
        # Add task-specific dynamics
        self.state += 0.05 * self.task_params * self.task_difficulty
        
        # Add noise
        self.state += np.random.randn(self.observation_dim) * 0.01
        
        self.time_step += 1
        
        # Get multi-modal observations
        observations = self.get_observation()
        
        # Compute reward (simple distance-based for demonstration)
        reward = -np.linalg.norm(self.state - self.task_params)
        
        # Episode completion
        done = self.time_step >= 100
        
        info = {
            "time_step": self.time_step,
            "current_task": self.current_task,
        }
        
        return observations, reward, done, info
        
    def get_observation(self):
        """
        Get current multi-modal observations.
        
        Returns:
            List of observation tensors, one per modality
        """
        observations = []
        
        for i in range(self.num_modalities):
            # Each modality sees a different "view" of the state
            view = self.state + np.random.randn(self.observation_dim) * 0.1 * i
            obs_tensor = torch.FloatTensor(view).unsqueeze(0)  # [1, obs_dim]
            observations.append(obs_tensor)
            
        return observations
        
    def change_task(self, task_id: int = None):
        """
        Change to a new task (for continual learning).
        
        Args:
            task_id: Optional task identifier, randomly generated if None
        """
        if task_id is not None:
            self.current_task = task_id
            np.random.seed(task_id)
        else:
            self.current_task += 1
            
        self.task_params = np.random.randn(self.observation_dim)
        self.reset()
