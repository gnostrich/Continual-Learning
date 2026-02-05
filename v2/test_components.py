"""Basic component tests for v2."""

import torch
from continual_learning import ControllerModel, EnvironmentPredictor, BaseEnvironment, DelayedObservationWrapper


def test_controller_forward():
    model = ControllerModel(input_dim=8, hidden_dim=16, action_dim=4, num_modalities=2)
    obs = [torch.randn(1, 8), torch.randn(1, 8)]
    action, internal = model(obs)
    assert action.shape == (1, 4)
    assert internal.shape == (1, 16)


def test_predictor_forward():
    predictor = EnvironmentPredictor(action_dim=4, hidden_dim=16, output_dim=8, num_modalities=2)
    action = torch.randn(1, 4)
    internal = torch.randn(1, 16)
    preds, pred_state = predictor(action, internal)
    assert len(preds) == 2
    assert preds[0].shape == (1, 8)
    assert pred_state.shape == (1, 16)


def test_environment_wrapper():
    env = BaseEnvironment(observation_dim=8, action_dim=4, num_modalities=2)
    wrapped = DelayedObservationWrapper(env, delay_steps=2)
    delayed, actual = wrapped.reset()
    assert actual is not None
    action = torch.randn(1, 4)
    delayed, actual, _, _, info = wrapped.step(action)
    assert "delay_steps" in info


def main():
    test_controller_forward()
    test_predictor_forward()
    test_environment_wrapper()
    print("v2 component tests passed")


if __name__ == "__main__":
    main()
