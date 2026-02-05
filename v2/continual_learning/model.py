"""Models for the asynchronous continual learning loop."""

import torch
import torch.nn as nn


class ControllerModel(nn.Module):
    """Controller that maps observations to actions with recurrent state."""

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 128,
        action_dim: int = 32,
        num_modalities: int = 2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.num_modalities = num_modalities

        self.modality_encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(num_modalities)
            ]
        )

        self.cross_modal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=4, batch_first=True
        )

        self.recurrent = nn.GRU(
            input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True
        )

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        self.hidden_state = None

    def reset_state(self):
        self.hidden_state = None

    def forward(self, observations, external_feedback=None):
        """
        Args:
            observations: list of tensors [batch, input_dim]
            external_feedback: optional tensor [batch, hidden_dim]

        Returns:
            action: [batch, action_dim]
            internal_state: [batch, hidden_dim]
        """
        batch_size = observations[0].size(0)
        device = observations[0].device

        encoded_modalities = []
        for idx, obs in enumerate(observations):
            encoded_modalities.append(self.modality_encoders[idx](obs))

        modality_stack = torch.stack(encoded_modalities, dim=1)
        attended, _ = self.cross_modal_attention(
            modality_stack, modality_stack, modality_stack
        )

        aggregated = attended.mean(dim=1, keepdim=True)
        if external_feedback is not None:
            aggregated = aggregated + external_feedback.unsqueeze(1)

        if self.hidden_state is None:
            self.hidden_state = torch.zeros(1, batch_size, self.hidden_dim, device=device)

        recurrent_out, self.hidden_state = self.recurrent(aggregated, self.hidden_state)
        internal_state = recurrent_out.squeeze(1)
        action = self.output_proj(internal_state)

        return action, internal_state


class EnvironmentPredictor(nn.Module):
    """Predicts environment response given action and controller state."""

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

        self.state_processor = nn.Sequential(
            nn.Linear(action_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.recurrent = nn.GRU(
            input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True
        )

        self.modality_decoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, output_dim),
                )
                for _ in range(num_modalities)
            ]
        )

        self.hidden_state = None

    def reset_state(self):
        self.hidden_state = None

    def forward(self, action, internal_state):
        batch_size = action.size(0)
        device = action.device

        combined = torch.cat([action, internal_state], dim=1)
        processed = self.state_processor(combined)

        if self.hidden_state is None:
            self.hidden_state = torch.zeros(1, batch_size, self.hidden_dim, device=device)

        recurrent_out, self.hidden_state = self.recurrent(
            processed.unsqueeze(1), self.hidden_state
        )
        pred_state = recurrent_out.squeeze(1)

        predictions = []
        for decoder in self.modality_decoders:
            predictions.append(decoder(pred_state))

        return predictions, pred_state
