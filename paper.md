# Continual Learning Loop: Self-Supervised Adaptation via Prediction Divergence

**Anonymous Authors**

## Abstract

We present a self-supervised continual learning system that learns environment dynamics without external labels or rewards. The system couples an action model (CrossModalModel) with an environment predictor (EnvironmentPredictor). The predictor forecasts the environment's response to the model's action; divergence between predicted and actual responses provides the training signal. This feedback loop enables continual adaptation while Elastic Weight Consolidation (EWC) mitigates catastrophic forgetting. A demo run produces per-task averages for loss, reward, and parameter divergence.

## 1. Introduction

Continual learning aims to adapt to new tasks without forgetting old ones. Most methods rely on labeled supervision or explicit reward signals. We propose a self-supervised alternative: learn environment dynamics by minimizing the mismatch between predicted and observed responses. This avoids external labels and uses prediction error as the learning signal.

## 2. Method

The system has two coupled models:

- **CrossModalModel**: Processes multi-modal observations and outputs an action
- **EnvironmentPredictor**: Predicts the environment's next multi-modal response given the action and internal state

Let $a_t$ be the action, $\hat{o}_{t+1}$ the predicted observation, and $o_{t+1}$ the true observation.

**Prediction divergence loss**:  
$\mathcal{L}_\text{div} = D$($\hat{o}_{t+1}$, $o_{t+1}$)

Total loss is a weighted sum:  
$\mathcal{L} = \mathcal{L}_\text{div} + \alpha \cdot \mathcal{L}_\text{action} + \beta \cdot \mathcal{L}_\text{EWC}$

## 3. Continual Learning Loop

For each task:
1. Observe multi-modal inputs
2. CrossModalModel outputs action
3. Environment returns new observation
4. EnvironmentPredictor predicts response
5. Minimize divergence + EWC to preserve past knowledge

EWC constrains parameter drift by penalizing deviation from previous task optima.

## 4. Experiments

The demo executes a 3-task continual learning cycle and logs:
- Per-task average loss
- Per-task average reward
- Parameter divergence across tasks

A summary plot (continual_learning_summary.png) visualizes these metrics. The run demonstrates stable divergence and adaptation without external labels.

## 5. Related Work

- **EWC** (Kirkpatrick et al.): Regularization-based forgetting mitigation
- **Predictive coding**: Learning via prediction error signals
- **World models**: Learning environment dynamics for control
- **Self-supervised RL**: Intrinsic signals without labels

Our approach combines predictive learning with EWC in a unified feedback loop.

## 6. Limitations

- Demo-only results; no large-scale benchmarks
- Divergence signal quality depends on environment realism
- No explicit comparison against other continual learning baselines

## 7. Future Work

- Benchmark against GEM, SI, LwF
- Scale to higher-dimensional environments
- Replace fixed divergence with adaptive or task-aware losses
- Add task detection or context inference

## 8. Conclusion

We present a self-supervised continual learning framework that trains solely on prediction divergence between expected and observed environment responses. By integrating an environment predictor with EWC regularization, the system adapts across tasks while mitigating forgetting.

**Code**: github.com/gnostrich/continual-learning  
**Demo outputs**: continual_learning_summary.png
