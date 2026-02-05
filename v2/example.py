"""Example run for v2 asynchronous continual learning."""

import os
import matplotlib.pyplot as plt
from continual_learning import (
    ControllerModel,
    EnvironmentPredictor,
    BaseEnvironment,
    DelayedObservationWrapper,
    ContinualLearningLoop,
)


def ensure_output_dir():
    output_dir = os.path.join("v2", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def main():
    output_dir = ensure_output_dir()

    model = ControllerModel(input_dim=64, hidden_dim=128, action_dim=32, num_modalities=2)
    predictor = EnvironmentPredictor(action_dim=32, hidden_dim=128, output_dim=64, num_modalities=2)
    env = BaseEnvironment(observation_dim=64, action_dim=32, num_modalities=2)
    env = DelayedObservationWrapper(env, delay_steps=2)

    loop = ContinualLearningLoop(
        model=model,
        predictor=predictor,
        environment=env,
        learning_rate=3e-4,
        ewc_lambda=0.5,
        feedback_weight=0.3,
        divergence_weight=0.5,
        action_l2_weight=1e-3,
        observation_blend=0.5,
    )

    metrics = loop.continual_learning_cycle(num_tasks=3, episodes_per_task=10, verbose=True)

    task_avg_losses = [sum(x) / len(x) for x in metrics["task_losses"]]
    task_avg_rewards = [sum(x) / len(x) for x in metrics["task_rewards"]]
    divergences = metrics["divergences"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle("Continual Learning Summary (v2)")

    axes[0].plot(task_avg_losses, marker="o", color="red")
    axes[0].set_title("Avg Loss per Task")
    axes[0].set_xlabel("Task")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(task_avg_rewards, marker="o", color="green")
    axes[1].set_title("Avg Reward per Task")
    axes[1].set_xlabel("Task")
    axes[1].set_ylabel("Reward")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(divergences, marker="o", color="blue")
    axes[2].set_title("Parameter Divergence")
    axes[2].set_xlabel("Task")
    axes[2].set_ylabel("Divergence")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    output_path = os.path.join(output_dir, "continual_learning_summary.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print("Saved summary plot to {}".format(output_path))


if __name__ == "__main__":
    main()
