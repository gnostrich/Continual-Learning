# Continual Learning Systems

A collection of continual learning implementations with recurrent feedback architectures, demonstrating different approaches to continuous learning from environmental interactions.

## Overview

This repository provides two complementary implementations of continual learning systems:

1. **Simple Demonstration** (`demo.py`): A straightforward CartPole-based demonstration using basic neural networks
2. **Advanced Framework** (`continual_learning/`): A comprehensive cross-modal system with Elastic Weight Consolidation (EWC)

## Installation

```bash
pip install -r requirements.txt
```

---

## 1. Simple Continual Learning Demonstration

A minimal continual learning system using PyTorch with recurrent feedback loops, demonstrating how neural networks can learn continuously from environmental interactions.

### Features

- **Simple Neural Networks**: Choose between feed-forward or GRU-based architectures
- **Interactive Environment**: Uses OpenAI Gym's CartPole environment for real-time interaction
- **Recurrent Learning Loop**: Outputs affect the environment, which feeds back as inputs
- **Real-time Visualizations**: Track learning dynamics, divergence measures, and input-output alignment
- **Online Learning**: Network updates continuously from each experience

### Usage

#### Basic Usage (Feed-forward Network)

```bash
python demo.py
```

#### Using GRU Network

```bash
python demo.py --network gru
```

#### Custom Training Parameters

```bash
python demo.py --network gru --episodes 200 --lr 0.005
```

#### Command Line Arguments

- `--network`: Network architecture (`feedforward` or `gru`, default: `feedforward`)
- `--episodes`: Number of training episodes (default: `100`)
- `--lr`: Learning rate (default: `0.01`)

### How It Works

#### Continual Learning Loop

1. **Network Output → Environment**: The neural network processes the current state and outputs an action
2. **Environment Response**: The action modifies the environment (CartPole game)
3. **Feedback Loop**: The new environment state feeds back as input to the network
4. **Online Learning**: The network updates immediately based on the reward signal
5. **Repeat**: This creates a continuous learning cycle

#### Visualizations

The system generates real-time visualizations showing:

1. **Loss/Divergence Over Time**: Tracks how prediction error changes during learning
2. **Cumulative Reward**: Shows the environment's response to network outputs
3. **Action Distribution**: Displays input-output alignment through action frequencies
4. **Prediction Confidence**: Monitors learning dynamics via output probabilities

Visualizations are saved to `continual_learning_results.png` after training.

### Files

- `networks.py`: Neural network architectures (FeedForward and GRU)
- `visualization.py`: Real-time visualization utilities
- `demo.py`: Main demonstration script with continual learning loop
- `test_components.py`: Unit tests for components
- `examples.py`: Example usage scenarios

### Neural Networks

**Feed-forward Network**:
- Input layer → Hidden layer (32 units, ReLU) → Hidden layer (32 units, ReLU) → Output layer
- Suitable for stateless decision-making

**GRU Network**:
- Input → GRU layer (32 hidden units) → Fully connected output
- Maintains hidden state for temporal patterns
- Better for sequential decision-making

### Example Output

```
============================================================
CONTINUAL LEARNING DEMONSTRATION
============================================================
Network Type: FEEDFORWARD
Episodes: 100
Learning Rate: 0.01
============================================================

Starting Continual Learning with FEEDFORWARD network
============================================================
Episode   0 | Steps: 12  | Reward:   12.0 | Avg(10):   12.0
Episode  10 | Steps: 18  | Reward:   18.0 | Avg(10):   15.5
Episode  20 | Steps: 45  | Reward:   45.0 | Avg(10):   32.8
...
============================================================
Training completed!
Total steps: 4532
Average reward: 52.34
Final 10 episodes average: 87.60
```

---

## 2. Advanced Cross-Modal Continual Learning Framework

A comprehensive continual learning system with a recurrent feedback architecture, supporting multi-modal inputs and advanced knowledge preservation techniques.

### Features

- **A single cross-modal model**: Processes multi-modal inputs with interaction between internal and external signals
- **Environmental interaction**: Outputs act on a simulated environment and feed back as inputs (external/internal)
- **Continual learning**: Dynamically adapts to new input streams, minimizing model divergence while preserving prior knowledge

### Architecture

#### Cross-Modal Model (`continual_learning/model.py`)
- Processes multiple input modalities simultaneously
- Uses cross-attention for modality interaction
- Maintains internal recurrent state (GRU-based memory)
- Integrates external feedback from environment
- Outputs actions that affect the environment

#### Environment (`continual_learning/environment.py`)
- Simulates a dynamic environment
- Receives actions from the model
- Returns multi-modal observations
- Supports task switching for continual learning scenarios

#### Continual Learning Loop (`continual_learning/loop.py`)
- Connects model and environment in a feedback loop
- Implements Elastic Weight Consolidation (EWC) for knowledge preservation
- Monitors parameter divergence to prevent catastrophic forgetting
- Enables dynamic adaptation to new tasks

### Usage

#### Basic Example

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

#### Running the Advanced Demo

```bash
python example.py
```

### Key Features

#### 1. Cross-Modal Processing
The model processes multiple modalities (e.g., visual, auditory) simultaneously:
- Separate encoders for each modality
- Cross-attention mechanism for modality interaction
- Fused representation for decision making

#### 2. Recurrent Feedback Architecture
- Model's internal state feeds back into next step
- External feedback from environment influences processing
- Enables temporal dependencies and memory

#### 3. Continual Learning
- **Dynamic Adaptation**: Learns from continuous data streams
- **Knowledge Preservation**: Uses EWC to retain important knowledge from previous tasks
- **Divergence Monitoring**: Tracks how much model parameters change
- **Task Switching**: Seamlessly transitions between different tasks

#### 4. EWC (Elastic Weight Consolidation)
- Computes Fisher Information Matrix to identify important parameters
- Adds regularization loss to prevent changes to critical weights
- Balances plasticity (learning new tasks) and stability (retaining old knowledge)

### Architecture Details

#### Information Flow
```
Multi-modal Observations → Encoders → Cross-Attention → 
  ↓
Recurrent Layer (with internal state) → Output → Actions
  ↑                                        ↓
External Feedback ←――――――――――――― Environment
```

#### Learning Process
1. Model receives multi-modal observations from environment
2. Processes through modality encoders and cross-attention
3. Recurrent layer maintains internal state
4. Generates actions that affect environment
5. Environment returns new observations and feedback
6. EWC regularization preserves important knowledge
7. Model adapts while minimizing divergence

### Extending the System

#### Custom Modalities
Add new input modalities by adjusting `num_modalities` parameter:

```python
model = CrossModalModel(num_modalities=3)  # e.g., visual, audio, text
environment = Environment(num_modalities=3)
```

#### Custom Environments
Subclass `Environment` to create domain-specific scenarios:

```python
class CustomEnvironment(Environment):
    def step(self, action):
        # Custom environment dynamics
        ...
        return observations, reward, done, info
```

#### Different Learning Objectives
Modify the `train_step` method to use custom loss functions:

```python
def custom_train_step(self, observations, target):
    output, internal_state = self.model(observations)
    loss = custom_loss_function(output, target)
    # ... rest of training
```

## License

See LICENSE file for details.
