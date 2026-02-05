#!/usr/bin/env python3
"""
Validation script for v2 asynchronous continual learning.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from continual_learning import (
    ControllerModel,
    EnvironmentPredictor,
    DelayedObservationWrapper,
    ContinualLearningLoop,
)


class DeterministicEnvironment:
    """Deterministic environment where response = action * 0.5."""

    def __init__(self, observation_dim=64, action_dim=32, num_modalities=2):
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.num_modalities = num_modalities
        self.current_task = 0

    def reset(self):
        return self.get_observation()

    def step(self, action):
        if isinstance(action, torch.Tensor):
            action_np = action.detach().cpu().numpy().flatten()
        else:
            action_np = action.flatten()

        if len(action_np) < self.observation_dim:
            obs = np.zeros(self.observation_dim)
            obs[: len(action_np)] = action_np * 0.5
        else:
            obs = action_np[: self.observation_dim] * 0.5

        observations = [torch.FloatTensor(obs).unsqueeze(0) for _ in range(self.num_modalities)]
        reward = -np.linalg.norm(obs)
        done = False
        info = {}
        return observations, reward, done, info

    def get_observation(self):
        return [torch.zeros(1, self.observation_dim) for _ in range(self.num_modalities)]

    def change_task(self, task_id: int = None):
        self.current_task = task_id if task_id is not None else self.current_task + 1


def ensure_output_dir():
    output_dir = os.path.join("v2", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def test_prediction_improvement():
    model = ControllerModel(input_dim=64, hidden_dim=128, action_dim=32, num_modalities=2)
    predictor = EnvironmentPredictor(action_dim=32, hidden_dim=128, output_dim=64, num_modalities=2)
    env = DeterministicEnvironment(observation_dim=64, action_dim=32, num_modalities=2)
    env = DelayedObservationWrapper(env, delay_steps=2)

    loop = ContinualLearningLoop(
        model=model,
        predictor=predictor,
        environment=env,
        learning_rate=1e-3,
        ewc_lambda=0.0,
        feedback_weight=0.3,
        divergence_weight=1.0,
        action_l2_weight=1e-4,
        observation_blend=0.5,
    )

    # Measure initial prediction error
    initial_errors = []
    model.reset_state()
    predictor.reset_state()

    for _ in range(30):
        delayed_obs, actual_obs = env.reset()
        observations = actual_obs
        action, internal_state = model(observations)
        predicted_obs, _ = predictor(action, internal_state)
        error = sum(
            torch.nn.functional.mse_loss(pred, actual).item()
            for pred, actual in zip(predicted_obs, actual_obs)
        )
        initial_errors.append(error)

    initial_mean = float(np.mean(initial_errors))
    print("Initial prediction error: {:.6f}".format(initial_mean))

    # Train
    for _ in range(10):
        loop.run_episode(max_steps=10, verbose=False)

    # Measure final prediction error
    final_errors = []
    model.reset_state()
    predictor.reset_state()

    for _ in range(30):
        delayed_obs, actual_obs = env.reset()
        observations = actual_obs
        action, internal_state = model(observations)
        predicted_obs, _ = predictor(action, internal_state)
        error = sum(
            torch.nn.functional.mse_loss(pred, actual).item()
            for pred, actual in zip(predicted_obs, actual_obs)
        )
        final_errors.append(error)

    final_mean = float(np.mean(final_errors))
    print("Final prediction error: {:.6f}".format(final_mean))

    improvement = (initial_mean - final_mean) / max(initial_mean, 1e-6) * 100
    print("Improvement: {:.1f}%".format(improvement))

    return final_mean < initial_mean


def test_learning_curve():
    output_dir = ensure_output_dir()

    model = ControllerModel(input_dim=64, hidden_dim=128, action_dim=32, num_modalities=2)
    predictor = EnvironmentPredictor(action_dim=32, hidden_dim=128, output_dim=64, num_modalities=2)
    env = DeterministicEnvironment(observation_dim=64, action_dim=32, num_modalities=2)
    env = DelayedObservationWrapper(env, delay_steps=2)

    loop = ContinualLearningLoop(
        model=model,
        predictor=predictor,
        environment=env,
        learning_rate=1e-3,
        ewc_lambda=0.0,
        feedback_weight=0.3,
        divergence_weight=1.0,
        action_l2_weight=1e-4,
        observation_blend=0.5,
    )

    prediction_errors = []
    for episode in range(20):
        loop.run_episode(max_steps=10, verbose=False)
        errors = []
        for _ in range(10):
            delayed_obs, actual_obs = env.reset()
            observations = actual_obs
            action, internal_state = model(observations)
            predicted_obs, _ = predictor(action, internal_state)
            error = sum(
                torch.nn.functional.mse_loss(pred, actual).item()
                for pred, actual in zip(predicted_obs, actual_obs)
            )
            errors.append(error)
        prediction_errors.append(float(np.mean(errors)))

    early_avg = float(np.mean(prediction_errors[:5]))
    late_avg = float(np.mean(prediction_errors[-5:]))

    plt.figure(figsize=(8, 5))
    plt.plot(prediction_errors, marker="o", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Prediction Error (MSE)")
    plt.title("Learning Curve (v2)")
    plt.grid(True, alpha=0.3)
    output_path = os.path.join(output_dir, "prediction_learning_curve.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print("Saved learning curve to {}".format(output_path))

    return late_avg < early_avg


def main():
    print("CONTINUAL LEARNING V2 VALIDATION")
    results = []
    results.append(test_prediction_improvement())
    results.append(test_learning_curve())

    passed = sum(1 for r in results if r)
    total = len(results)
    print("Tests passed: {}/{}".format(passed, total))
    if passed == total:
        print("All tests passed")
    else:
        print("Some tests failed")


if __name__ == "__main__":
    main()
