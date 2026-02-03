# Continual Learning Demonstration

A minimal continual learning system using PyTorch with recurrent feedback loops, demonstrating how neural networks can learn continuously from environmental interactions.

## Features

- **Simple Neural Networks**: Choose between feed-forward or GRU-based architectures
- **Interactive Environment**: Uses OpenAI Gym's CartPole environment for real-time interaction
- **Recurrent Learning Loop**: Outputs affect the environment, which feeds back as inputs
- **Real-time Visualizations**: Track learning dynamics, divergence measures, and input-output alignment
- **Online Learning**: Network updates continuously from each experience

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage (Feed-forward Network)

```bash
python demo.py
```

### Using GRU Network

```bash
python demo.py --network gru
```

### Custom Training Parameters

```bash
python demo.py --network gru --episodes 200 --lr 0.005
```

### Command Line Arguments

- `--network`: Network architecture (`feedforward` or `gru`, default: `feedforward`)
- `--episodes`: Number of training episodes (default: `100`)
- `--lr`: Learning rate (default: `0.01`)

## How It Works

### Continual Learning Loop

1. **Network Output → Environment**: The neural network processes the current state and outputs an action
2. **Environment Response**: The action modifies the environment (CartPole game)
3. **Feedback Loop**: The new environment state feeds back as input to the network
4. **Online Learning**: The network updates immediately based on the reward signal
5. **Repeat**: This creates a continuous learning cycle

### Visualizations

The system generates real-time visualizations showing:

1. **Loss/Divergence Over Time**: Tracks how prediction error changes during learning
2. **Cumulative Reward**: Shows the environment's response to network outputs
3. **Action Distribution**: Displays input-output alignment through action frequencies
4. **Prediction Confidence**: Monitors learning dynamics via output probabilities

Visualizations are saved to `continual_learning_results.png` after training.

## Architecture

### Files

- `networks.py`: Neural network architectures (FeedForward and GRU)
- `visualization.py`: Real-time visualization utilities
- `demo.py`: Main demonstration script with continual learning loop
- `requirements.txt`: Python dependencies

### Neural Networks

**Feed-forward Network**:
- Input layer → Hidden layer (32 units, ReLU) → Hidden layer (32 units, ReLU) → Output layer
- Suitable for stateless decision-making

**GRU Network**:
- Input → GRU layer (32 hidden units) → Fully connected output
- Maintains hidden state for temporal patterns
- Better for sequential decision-making

## Example Output

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

## License

See LICENSE file for details.
