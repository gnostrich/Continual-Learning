"""
Simple neural network architectures for continual learning demonstration.
"""
import torch
import torch.nn as nn


class FeedForwardNetwork(nn.Module):
    """Simple feed-forward neural network."""
    
    def __init__(self, input_size, hidden_size, output_size):
        super(FeedForwardNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.network(x)


class GRUNetwork(nn.Module):
    """GRU-based recurrent neural network."""
    
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden=None):
        """
        Forward pass through GRU network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            hidden: Optional hidden state
            
        Returns:
            output: Network output
            hidden: Updated hidden state
        """
        if len(x.shape) == 1:
            x = x.unsqueeze(0).unsqueeze(0)  # Add batch and sequence dimensions
        elif len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
        
        gru_out, hidden = self.gru(x, hidden)
        output = self.fc(gru_out[:, -1, :])  # Use last output
        return output, hidden
    
    def init_hidden(self, batch_size=1):
        """Initialize hidden state."""
        return torch.zeros(1, batch_size, self.hidden_size)
