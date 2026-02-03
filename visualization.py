"""
Visualization utilities for continual learning demonstration.
"""
import matplotlib.pyplot as plt
import numpy as np
from collections import deque


class ContinualLearningVisualizer:
    """Manages visualizations for continual learning metrics."""
    
    def __init__(self, max_history=500):
        self.max_history = max_history
        self.losses = deque(maxlen=max_history)
        self.rewards = deque(maxlen=max_history)
        self.actions = deque(maxlen=max_history)
        self.predictions = deque(maxlen=max_history)
        
        # Setup the figure
        plt.ion()  # Enable interactive mode
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle('Continual Learning Dynamics', fontsize=14, fontweight='bold')
    
    def update(self, loss, reward, action, prediction):
        """Update metrics history."""
        self.losses.append(loss)
        self.rewards.append(reward)
        self.actions.append(action)
        self.predictions.append(prediction)
    
    def plot(self, episode):
        """Generate visualizations of learning dynamics."""
        # Clear all axes
        for ax in self.axes.flat:
            ax.clear()
        
        # Plot 1: Loss over time (Divergence measure)
        ax1 = self.axes[0, 0]
        if len(self.losses) > 0:
            ax1.plot(list(self.losses), color='red', alpha=0.7, linewidth=1)
            ax1.set_title('Loss/Divergence Over Time')
            ax1.set_xlabel('Step')
            ax1.set_ylabel('Loss')
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: Rewards over time (Environment feedback)
        ax2 = self.axes[0, 1]
        if len(self.rewards) > 0:
            ax2.plot(list(self.rewards), color='green', alpha=0.7, linewidth=1)
            ax2.set_title('Cumulative Reward (Environment Response)')
            ax2.set_xlabel('Step')
            ax2.set_ylabel('Cumulative Reward')
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Action distribution (Input-Output alignment)
        ax3 = self.axes[1, 0]
        if len(self.actions) > 0:
            actions_array = np.array(list(self.actions))
            unique, counts = np.unique(actions_array, return_counts=True)
            ax3.bar(unique, counts, color='blue', alpha=0.7)
            ax3.set_title('Action Distribution (Output Alignment)')
            ax3.set_xlabel('Action')
            ax3.set_ylabel('Frequency')
            ax3.set_xticks(unique)
            ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Prediction confidence over time
        ax4 = self.axes[1, 1]
        if len(self.predictions) > 0:
            predictions_array = np.array(list(self.predictions))
            if len(predictions_array.shape) == 2:
                # Plot max confidence (prediction strength)
                max_probs = np.max(predictions_array, axis=1)
                ax4.plot(max_probs, color='purple', alpha=0.7, linewidth=1)
                ax4.set_title('Prediction Confidence (Learning Dynamics)')
                ax4.set_xlabel('Step')
                ax4.set_ylabel('Max Action Probability')
                ax4.set_ylim([0, 1])
                ax4.grid(True, alpha=0.3)
        
        self.fig.tight_layout()
        plt.pause(0.01)  # Brief pause to update display
    
    def save(self, filename='continual_learning_results.png'):
        """Save the current visualization to a file."""
        self.fig.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {filename}")
    
    def close(self):
        """Close the visualization."""
        plt.ioff()
        plt.close(self.fig)
