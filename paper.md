# Continual Learning Loop v2: Self-Supervised Adaptation with Asynchronous Observations

**Anonymous Authors**

## Abstract

We present a self-supervised continual learning system that explicitly models asynchrony between actions and observations. The system couples a controller with an environment predictor. The predictor forecasts the environment response to the controller action, and divergence between predicted and actual responses provides the shared learning signal. A delayed-observation wrapper exposes the controller to stale signals while blending in predicted estimates, making the loop robust to latency and partial observability. Elastic Weight Consolidation (EWC) is retained to mitigate catastrophic forgetting across tasks.

## 1. Introduction

Continual learning aims to adapt to new tasks without forgetting old ones. Many methods rely on labeled supervision or explicit rewards. We propose a self-supervised alternative that learns environment dynamics through prediction error while acknowledging asynchronous feedback: the system can act even when the latest observation has not arrived. This matches settings where outcomes are delayed or noisy.

## 2. Method

The system has two coupled models:

- **ControllerModel**: processes multi-modal observations and outputs an action
- **EnvironmentPredictor**: predicts the next multi-modal observation given the action and controller state

Let $a_t$ be the action, $\hat{o}_{t+1}$ the predicted observation, and $o_{t+1}$ the true observation.

**Prediction divergence loss**:
$\mathcal{L}_\text{div} = D(\hat{o}_{t+1}, o_{t+1})$

Total loss is a weighted sum:
$\mathcal{L} = \mathcal{L}_\text{div} + \alpha \cdot \mathcal{L}_\text{action} + \beta \cdot \mathcal{L}_\text{EWC}$

### Asynchronous Observation Handling

We introduce a delayed-observation wrapper that returns both delayed and actual observations. The controller can consume a blended input:

$\tilde{o}_t = (1 - \gamma) \cdot o^{\text{delayed}}_t + \gamma \cdot \hat{o}_t$

where $\hat{o}_t$ is the predictor estimate produced from the previous action and state. This allows the controller to act under stale feedback while still grounding updates in the actual observation when it arrives.

## 3. Continual Learning Loop

For each task:
1. Receive delayed observation (or prediction when delay is active)
2. Controller outputs action
3. Environment returns actual response (possibly later)
4. Predictor forecasts response from action + state
5. Minimize divergence + EWC to preserve past knowledge

EWC constrains parameter drift by penalizing deviation from previous task optima.

## 4. Experiments

The demo executes a multi-task continual learning cycle and logs:
- Per-task average loss
- Per-task average reward
- Parameter divergence across tasks

Outputs are saved under v2/outputs/.

## 5. Related Work

- **EWC** (Kirkpatrick et al.): Regularization-based forgetting mitigation
- **Predictive coding**: Learning via prediction error signals
- **World models**: Learning environment dynamics for control
- **Self-supervised RL**: Intrinsic signals without labels

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

We present a self-supervised continual learning framework that trains on prediction divergence between expected and observed environment responses while explicitly handling asynchronous observations. By integrating a delayed-observation wrapper and a predictor with EWC regularization, the system adapts across tasks while mitigating forgetting.

## V1 Note

The original paper and implementation are archived under v1/. The v2 changes focus on explicit asynchronicity handling, blended observations for the controller, and updated validation/output paths.
