"""Cross-modal model with internal and external signal interaction."""

import torch
import torch.nn as nn


class CrossModalModel(nn.Module):
    """
    A single cross-modal model that processes multi-modal inputs.
    
    The model integrates:
    - External inputs (from environment observations)
    - Internal signals (recurrent state/memory)
    - Cross-modal interaction through attention mechanisms
    """
    
    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 32,
        num_modalities: int = 2,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_modalities = num_modalities
        
        # Modality-specific encoders
        self.modality_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            for _ in range(num_modalities)
        ])
        
        # Cross-modal attention for interaction between modalities
        self.cross_modal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True
        )
        
        # Recurrent layer for internal state (memory)
        self.recurrent = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Internal state (memory)
        self.hidden_state = None
        
    def reset_state(self):
        """Reset internal hidden state."""
        self.hidden_state = None
        
    def forward(self, inputs, external_feedback=None):
        """
        Forward pass with multi-modal inputs and optional external feedback.
        
        Args:
            inputs: List of tensors, one per modality [batch, input_dim]
            external_feedback: Optional tensor from environment [batch, hidden_dim]
            
        Returns:
            output: Action/output tensor [batch, output_dim]
            internal_state: Current internal state for feedback [batch, hidden_dim]
        """
        batch_size = inputs[0].size(0)
        
        # Encode each modality
        encoded_modalities = []
        for i, modality_input in enumerate(inputs):
            encoded = self.modality_encoders[i](modality_input)
            encoded_modalities.append(encoded)
        
        # Stack modalities for cross-modal attention [batch, num_modalities, hidden_dim]
        modality_stack = torch.stack(encoded_modalities, dim=1)
        
        # Apply cross-modal attention (interaction between modalities)
        attended, _ = self.cross_modal_attention(
            modality_stack, modality_stack, modality_stack
        )
        
        # Aggregate across modalities
        aggregated = attended.mean(dim=1, keepdim=True)  # [batch, 1, hidden_dim]
        
        # Incorporate external feedback if provided
        if external_feedback is not None:
            aggregated = aggregated + external_feedback.unsqueeze(1)
        
        # Process through recurrent layer (internal state)
        if self.hidden_state is None:
            self.hidden_state = torch.zeros(1, batch_size, self.hidden_dim)
            if inputs[0].is_cuda:
                self.hidden_state = self.hidden_state.cuda()
        
        recurrent_out, self.hidden_state = self.recurrent(
            aggregated, self.hidden_state
        )
        
        # Generate output
        output = self.output_proj(recurrent_out.squeeze(1))
        internal_state = recurrent_out.squeeze(1)
        
        return output, internal_state


class EnvironmentPredictor(nn.Module):
    """
    Predictor model that learns to forecast environment responses.
    
    Given an action and current internal state, predicts what the environment
    will return (observation/feedback). Used for self-supervised continual learning
    via divergence minimization between predicted and actual responses.
    """
    
    def __init__(
        self,
        action_dim: int = 32,
        hidden_dim: int = 128,
        output_dim: int = 64,
        num_modalities: int = 2,
    ):
        super().__init__()
        
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_modalities = num_modalities
        
        # Process action and internal state
        self.state_processor = nn.Sequential(
            nn.Linear(action_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Recurrent processing of dynamics
        self.recurrent = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        
        # Multi-modal output decoders (predict each modality separately)
        self.modality_decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
            for _ in range(num_modalities)
        ])
        
        self.hidden_state = None
    
    def reset_state(self):
        """Reset internal hidden state."""
        self.hidden_state = None
    
    def forward(self, action, internal_state):
        """
        Predict environment response given action and current state.
        
        Args:
            action: Action tensor [batch, action_dim]
            internal_state: Internal state tensor [batch, hidden_dim]
            
        Returns:
            predictions: List of predicted observations (one per modality)
            pred_state: Predicted internal state for next step
        """
        batch_size = action.size(0)
        
        # Combine action and internal state
        combined = torch.cat([action, internal_state], dim=1)
        processed = self.state_processor(combined)
        
        # Process through recurrent layer
        if self.hidden_state is None:
            self.hidden_state = torch.zeros(1, batch_size, self.hidden_dim)
            if action.is_cuda:
                self.hidden_state = self.hidden_state.cuda()
        
        recurrent_out, self.hidden_state = self.recurrent(
            processed.unsqueeze(1), self.hidden_state
        )
        
        pred_state = recurrent_out.squeeze(1)
        
        # Decode predictions for each modality
        predictions = []
        for decoder in self.modality_decoders:
            pred = decoder(pred_state)
            predictions.append(pred)
        
        return predictions, pred_state
