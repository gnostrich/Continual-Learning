"""Continual learning loop with recurrent feedback architecture."""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Dict, List
import copy


class ContinualLearningLoop:
    """
    Main continual learning loop that integrates model, environment, and learning.
    
    Features:
    - Recurrent feedback: Model outputs feed back as environment inputs
    - Dynamic adaptation: Continuously learns from new data streams
    - Knowledge preservation: Uses EWC (Elastic Weight Consolidation) to prevent forgetting
    - Divergence monitoring: Tracks model parameter changes
    """
    
    def __init__(
        self,
        model,
        environment,
        learning_rate: float = 1e-3,
        ewc_lambda: float = 0.5,
        feedback_weight: float = 0.3,
    ):
        self.model = model
        self.environment = environment
        self.learning_rate = learning_rate
        self.ewc_lambda = ewc_lambda
        self.feedback_weight = feedback_weight
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # For knowledge preservation (EWC)
        self.fisher_information = {}
        self.optimal_params = {}
        self.task_count = 0
        
        # Metrics tracking
        self.metrics = {
            "losses": [],
            "rewards": [],
            "divergence": [],
        }
        
    def compute_fisher_information(self, num_samples: int = 100):
        """
        Compute Fisher Information Matrix for EWC.
        
        This estimates the importance of each parameter for the current task.
        """
        self.model.eval()
        fisher = {}
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher[name] = torch.zeros_like(param)
        
        # Sample interactions to estimate Fisher information
        for _ in range(num_samples):
            self.model.zero_grad()
            self.model.reset_state()  # Reset internal state
            
            # Get observation from environment
            observations = self.environment.get_observation()
            
            # Forward pass
            output, internal_state = self.model(observations)
            
            # Compute log probability (simplified)
            log_prob = -torch.norm(output, p=2)
            log_prob.backward()
            
            # Accumulate squared gradients
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.detach() ** 2
        
        # Average over samples
        for name in fisher:
            fisher[name] /= num_samples
            
        self.fisher_information = fisher
        
    def save_optimal_params(self):
        """Save current parameters as optimal for the current task."""
        self.optimal_params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.optimal_params[name] = param.data.clone()
                
    def ewc_loss(self):
        """
        Compute EWC regularization loss.
        
        Penalizes changes to important parameters from previous tasks.
        """
        if not self.fisher_information:
            return torch.tensor(0.0)
        
        loss = torch.tensor(0.0)
        for name, param in self.model.named_parameters():
            if name in self.fisher_information and param.requires_grad:
                fisher = self.fisher_information[name]
                optimal = self.optimal_params[name]
                loss += (fisher * (param - optimal) ** 2).sum()
                
        return loss
        
    def compute_divergence(self):
        """
        Compute divergence between current and optimal parameters.
        
        Returns average L2 distance of parameters.
        """
        if not self.optimal_params:
            return 0.0
        
        total_divergence = 0.0
        num_params = 0
        
        for name, param in self.model.named_parameters():
            if name in self.optimal_params and param.requires_grad:
                optimal = self.optimal_params[name]
                divergence = torch.norm(param - optimal, p=2).item()
                total_divergence += divergence
                num_params += 1
                
        return total_divergence / max(num_params, 1)
        
    def train_step(self, observations, target=None, external_feedback=None):
        """
        Single training step with continual learning.
        
        Args:
            observations: List of observation tensors (multi-modal)
            target: Optional target for supervised learning
            external_feedback: Optional external feedback signal
            
        Returns:
            loss: Total loss value
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        output, internal_state = self.model(observations, external_feedback)
        
        # Task loss (can be supervised or unsupervised)
        if target is not None:
            task_loss = nn.functional.mse_loss(output, target)
        else:
            # Unsupervised: minimize output variance (stability)
            task_loss = output.var()
        
        # EWC regularization for knowledge preservation
        ewc_reg = self.ewc_loss()
        
        # Total loss
        total_loss = task_loss + self.ewc_lambda * ewc_reg
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return total_loss.item()
        
    def run_episode(self, max_steps: int = 100, verbose: bool = False):
        """
        Run a single episode with recurrent feedback.
        
        Args:
            max_steps: Maximum steps per episode
            verbose: Whether to print progress
            
        Returns:
            episode_metrics: Dict with episode statistics
        """
        observations = self.environment.reset()
        self.model.reset_state()
        
        episode_losses = []
        episode_rewards = []
        internal_state = None
        
        for step in range(max_steps):
            # Use internal state as external feedback (recurrent architecture)
            external_feedback = None
            if internal_state is not None:
                external_feedback = internal_state * self.feedback_weight
            
            # Training step
            loss = self.train_step(observations, external_feedback=external_feedback)
            episode_losses.append(loss)
            
            # Get action from model
            with torch.no_grad():
                action, internal_state = self.model(observations, external_feedback)
            
            # Environment step
            observations, reward, done, info = self.environment.step(action)
            episode_rewards.append(reward)
            
            if done:
                break
                
        # Episode metrics
        metrics = {
            "mean_loss": sum(episode_losses) / len(episode_losses),
            "total_reward": sum(episode_rewards),
            "steps": len(episode_losses),
        }
        
        if verbose:
            print(f"Episode: Loss={metrics['mean_loss']:.4f}, "
                  f"Reward={metrics['total_reward']:.2f}, "
                  f"Steps={metrics['steps']}")
        
        return metrics
        
    def continual_learning_cycle(
        self,
        num_tasks: int = 3,
        episodes_per_task: int = 10,
        verbose: bool = True,
    ):
        """
        Run full continual learning cycle across multiple tasks.
        
        Args:
            num_tasks: Number of tasks to learn sequentially
            episodes_per_task: Episodes per task
            verbose: Whether to print progress
            
        Returns:
            all_metrics: Dict with metrics across all tasks
        """
        all_metrics = {
            "task_losses": [],
            "task_rewards": [],
            "divergences": [],
        }
        
        for task_id in range(num_tasks):
            if verbose:
                print(f"\n=== Task {task_id + 1}/{num_tasks} ===")
            
            # Change to new task
            self.environment.change_task(task_id)
            
            task_losses = []
            task_rewards = []
            
            # Train on current task
            for episode in range(episodes_per_task):
                metrics = self.run_episode(verbose=verbose and episode % 5 == 0)
                task_losses.append(metrics["mean_loss"])
                task_rewards.append(metrics["total_reward"])
            
            # After task completion, compute Fisher information and save params
            if verbose:
                print(f"Computing Fisher information for task {task_id + 1}...")
            self.compute_fisher_information(num_samples=50)
            self.save_optimal_params()
            
            # Track divergence
            divergence = self.compute_divergence()
            
            all_metrics["task_losses"].append(task_losses)
            all_metrics["task_rewards"].append(task_rewards)
            all_metrics["divergences"].append(divergence)
            
            if verbose:
                print(f"Task {task_id + 1} complete. "
                      f"Avg Loss: {sum(task_losses)/len(task_losses):.4f}, "
                      f"Avg Reward: {sum(task_rewards)/len(task_rewards):.2f}, "
                      f"Divergence: {divergence:.4f}")
            
            self.task_count += 1
        
        return all_metrics
