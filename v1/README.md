# Continual Learning Loop

A self-supervised continual learning system with predictor-based divergence minimization. The system learns environment dynamics without external supervision by using two models:

- **Model 1 (Action Model)**: Cross-modal model that processes observations and outputs actions
- **Model 2 (Environment Predictor)**: Learns to predict what the environment will respond with
- **Self-supervised learning**: Trains by minimizing divergence between predicted and actual environment responses
- **Continual adaptation**: Learns purely from prediction errors—no external labels or rewards required

## Motivation

Traditional continual learning requires external supervision (labels/rewards). This system implements **truly self-supervised continual learning**:

1. **Model 1** outputs an action based on current observations
2. **Model 2** predicts what the environment will respond with
3. Environment provides actual response
4. **Both models learn** by minimizing divergence between predicted vs actual

The prediction error becomes the learning signal—no external training data needed. The system learns environment dynamics purely from its own prediction mistakes.

## Architecture

### Model 1: Cross-Modal Action Model (`continual_learning/model.py`)
- Processes multiple input modalities simultaneously
- Uses cross-attention for modality interaction
- Maintains internal recurrent state (GRU-based memory)
- Integrates external feedback from environment
- **Outputs actions** that affect the environment
- 235,424 parameters

### Model 2: Environment Predictor (`continual_learning/model.py`)
- **Key innovation for self-supervised learning**
- Takes action + internal state as input
- **Predicts environment's multi-modal response**
- Trained jointly with action model via divergence loss
- Enables learning without external labels
- 185,728 parameters

### Environment (`continual_learning/environment.py`)
- Simulates a dynamic environment
- Receives actions from Model 1
- Returns multi-modal observations
- Supports task switching for continual learning scenarios

### Continual Learning Loop (`continual_learning/loop.py`)
- Connects both models and environment in feedback loop
- **Trains via prediction divergence minimization**
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
from continual_learning import CrossModalModel, EnvironmentPredictor, Environment, ContinualLearningLoop

# Initialize Model 1 (Action Model)
model = CrossModalModel(
    input_dim=64,
    hidden_dim=128,
    output_dim=32,
    num_modalities=2,
)

# Initialize Model 2 (Environment Predictor)
predictor = EnvironmentPredictor(
    action_dim=32,
    hidden_dim=128,
    output_dim=64,
    num_modalities=2,
)

# Initialize Environment
environment = Environment(
    observation_dim=64,
    action_dim=32,
    num_modalities=2,
)

# Create learning loop (trains both models via divergence)
learning_loop = ContinualLearningLoop(
    model=model,
    predictor=predictor,
    environment=environment,
    learning_rate=3e-4,
    ewc_lambda=0.5,  # Knowledge preservation strength
    feedback_weight=0.3,  # Recurrent feedback weight
    divergence_weight=0.5,  # Prediction divergence loss weight
    action_l2_weight=1e-3,  # Action regularization
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

### 3. Self-Supervised Continual Learning
- **No external labels required**: Learns purely from prediction errors
- **Prediction divergence as loss**: Minimizes difference between predicted and actual environment responses
- **Dynamic Adaptation**: Learns environment dynamics from continuous interaction
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
                    ┌──────────────────────┐
                    │   Model 1 (Action)   │
Observations ──────>│  Cross-Modal Model   │────> Action
                    └──────────────────────┘       │
                              │                     │
                         Internal State            │
                              │                     ▼
                              ▼              ┌─────────────┐
                    ┌──────────────────────┐ │ Environment │
                    │ Model 2 (Predictor)  │ └─────────────┘
                    │ Environment Dynamics │        │
                    └──────────────────────┘        │
                              │                     │
                      Predicted Response      Actual Response
                              │                     │
                              └─────> Compare <─────┘
                                   (Divergence Loss)
```

### Learning Process
1. **Model 1** receives multi-modal observations from environment
2. Processes through modality encoders and cross-attention
3. Recurrent layer maintains internal state
4. **Generates action** that affects environment
5. **Model 2** predicts what environment will respond with (using action + state)
6. **Environment** returns actual response
7. **Compute divergence** between predicted and actual response
8. **Both models update** to minimize prediction error
9. EWC regularization preserves important knowledge
10. System learns environment dynamics through self-supervision

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

### Validation
Run the validation suite to verify the system learns correctly:

```bash
python validate_predictor.py
```

This tests:
- Prediction accuracy improvement over training
- Learning curve analysis
- Environment dynamics learning

Expected result: Prediction error should decrease significantly (>99% improvement on deterministic environments).

## License

See LICENSE file for details.
