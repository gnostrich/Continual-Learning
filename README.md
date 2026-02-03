# Continual Learning Loop

A minimal continual learning system with a recurrent feedback architecture. The system integrates:

- **A single cross-modal model**: Processes multi-modal inputs with interaction between internal and external signals
- **Environmental interaction**: Outputs act on a simulated environment and feed back as inputs (external/internal)
- **Continual learning**: Dynamically adapts to new input streams, minimizing model divergence while preserving prior knowledge

## Architecture

### Cross-Modal Model (`continual_learning/model.py`)
- Processes multiple input modalities simultaneously
- Uses cross-attention for modality interaction
- Maintains internal recurrent state (GRU-based memory)
- Integrates external feedback from environment
- Outputs actions that affect the environment

### Environment (`continual_learning/environment.py`)
- Simulates a dynamic environment
- Receives actions from the model
- Returns multi-modal observations
- Supports task switching for continual learning scenarios

### Continual Learning Loop (`continual_learning/loop.py`)
- Connects model and environment in a feedback loop
- Implements Elastic Weight Consolidation (EWC) for knowledge preservation
- Monitors parameter divergence to prevent catastrophic forgetting
- Enables dynamic adaptation to new tasks

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Example

```python
from continual_learning import CrossModalModel, Environment, ContinualLearningLoop

# Initialize components
model = CrossModalModel(
    input_dim=64,
    hidden_dim=128,
    output_dim=32,
    num_modalities=2,
)

environment = Environment(
    observation_dim=64,
    action_dim=32,
    num_modalities=2,
)

# Create learning loop
learning_loop = ContinualLearningLoop(
    model=model,
    environment=environment,
    learning_rate=1e-3,
    ewc_lambda=0.5,  # Knowledge preservation strength
    feedback_weight=0.3,  # Recurrent feedback weight
)

# Run continual learning across multiple tasks
metrics = learning_loop.continual_learning_cycle(
    num_tasks=3,
    episodes_per_task=10,
    verbose=True,
)
```

### Running the Demo

```bash
python example.py
```

## Key Features

### 1. Cross-Modal Processing
The model processes multiple modalities (e.g., visual, auditory) simultaneously:
- Separate encoders for each modality
- Cross-attention mechanism for modality interaction
- Fused representation for decision making

### 2. Recurrent Feedback Architecture
- Model's internal state feeds back into next step
- External feedback from environment influences processing
- Enables temporal dependencies and memory

### 3. Continual Learning
- **Dynamic Adaptation**: Learns from continuous data streams
- **Knowledge Preservation**: Uses EWC to retain important knowledge from previous tasks
- **Divergence Monitoring**: Tracks how much model parameters change
- **Task Switching**: Seamlessly transitions between different tasks

### 4. EWC (Elastic Weight Consolidation)
- Computes Fisher Information Matrix to identify important parameters
- Adds regularization loss to prevent changes to critical weights
- Balances plasticity (learning new tasks) and stability (retaining old knowledge)

## Architecture Details

### Information Flow
```
Multi-modal Observations → Encoders → Cross-Attention → 
  ↓
Recurrent Layer (with internal state) → Output → Actions
  ↑                                        ↓
External Feedback ←――――――――――――― Environment
```

### Learning Process
1. Model receives multi-modal observations from environment
2. Processes through modality encoders and cross-attention
3. Recurrent layer maintains internal state
4. Generates actions that affect environment
5. Environment returns new observations and feedback
6. EWC regularization preserves important knowledge
7. Model adapts while minimizing divergence

## Extending the System

### Custom Modalities
Add new input modalities by adjusting `num_modalities` parameter:

```python
model = CrossModalModel(num_modalities=3)  # e.g., visual, audio, text
environment = Environment(num_modalities=3)
```

### Custom Environments
Subclass `Environment` to create domain-specific scenarios:

```python
class CustomEnvironment(Environment):
    def step(self, action):
        # Custom environment dynamics
        ...
        return observations, reward, done, info
```

### Different Learning Objectives
Modify the `train_step` method to use custom loss functions:

```python
def custom_train_step(self, observations, target):
    output, internal_state = self.model(observations)
    loss = custom_loss_function(output, target)
    # ... rest of training
```

## License

See LICENSE file for details.
