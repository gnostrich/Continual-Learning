# Continual Learning v2 (Asynchronous World Model)

This version makes the temporal/asynchronous setting explicit:

- The controller acts on delayed or partial observations.
- The environment predictor estimates what the environment will return.
- The prediction error provides a shared learning signal.

The predictor adds the most value when observations are delayed or noisy,
so this v2 loop uses a delayed-observation wrapper and blends predicted
and delayed signals for controller input.

## Key Ideas

- Controller (policy): produces actions from observations and internal state.
- Environment predictor (world model): forecasts the next observation from action + state.
- Shared loss: divergence between predicted and actual observations updates both models.
- Asynchronous environment: controller may receive stale observations; predictor fills the gap.

## Files

- continual_learning/model.py: controller and predictor models
- continual_learning/environment.py: base environment and delayed wrapper
- continual_learning/loop.py: training loop with asynchronous handling
- example.py: demo run and summary plot
- validate_predictor.py: idiot check validation
- test_components.py: basic component tests

## Running

```bash
docker run --rm -v $(pwd):/workspace -w /workspace python:3.10 \
  bash -lc "pip install -r v2/requirements.txt && python v2/example.py"
```

Outputs:

- v2/outputs/continual_learning_summary.png
- v2/outputs/prediction_learning_curve.png

## Notes

This v2 design is implementation-agnostic and focuses on the theoretical loop:
controller actions, predictor forecasts, and divergence-driven learning under
asynchronous observation.
