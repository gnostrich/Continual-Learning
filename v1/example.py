"""Example usage of the continual learning system."""

import torch
import matplotlib.pyplot as plt
from continual_learning import CrossModalModel, EnvironmentPredictor, Environment, ContinualLearningLoop


def main():
    """Demonstrate the continual learning system with recurrent feedback."""
    
    print("=" * 60)
    print("Continual Learning System - Demo")
    print("=" * 60)
    
    # Configuration
    input_dim = 64
    hidden_dim = 128
    output_dim = 32
    num_modalities = 2
    
    print(f"\nConfiguration:")
    print(f"  Input dimension: {input_dim}")
    print(f"  Hidden dimension: {hidden_dim}")
    print(f"  Output dimension: {output_dim}")
    print(f"  Number of modalities: {num_modalities}")
    
    # Initialize components
    print("\n" + "=" * 60)
    print("Initializing Components")
    print("=" * 60)
    
    # 1. Cross-modal model
    model = CrossModalModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_modalities=num_modalities,
    )
    print(f"\n✓ Cross-modal model initialized")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 2. Environment predictor (self-supervised dynamics model)
    predictor = EnvironmentPredictor(
        action_dim=output_dim,
        hidden_dim=hidden_dim,
        output_dim=input_dim,
        num_modalities=num_modalities,
    )
    print(f"\n✓ Environment predictor initialized")
    print(f"  Parameters: {sum(p.numel() for p in predictor.parameters()):,}")

    # 3. Simulated environment
    environment = Environment(
        observation_dim=input_dim,
        action_dim=output_dim,
        num_modalities=num_modalities,
        task_difficulty=0.5,
    )
    print(f"\n✓ Environment initialized")
    print(f"  Observation dim: {environment.observation_dim}")
    print(f"  Action dim: {environment.action_dim}")
    
    # 4. Continual learning loop
    learning_loop = ContinualLearningLoop(
        model=model,
        predictor=predictor,
        environment=environment,
        learning_rate=3e-4,
        ewc_lambda=0.5,  # Regularization strength for knowledge preservation
        feedback_weight=0.3,  # Weight for recurrent feedback
        divergence_weight=0.5,  # Scale for prediction divergence
        action_l2_weight=1e-3,  # Action regularization for stability
    )
    print(f"\n✓ Continual learning loop initialized")
    print(f"  Learning rate: {learning_loop.learning_rate}")
    print(f"  EWC lambda: {learning_loop.ewc_lambda}")
    print(f"  Feedback weight: {learning_loop.feedback_weight}")
    
    # Run continual learning cycle
    print("\n" + "=" * 60)
    print("Running Continual Learning Cycle")
    print("=" * 60)
    
    metrics = learning_loop.continual_learning_cycle(
        num_tasks=3,
        episodes_per_task=10,
        verbose=True,
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    for i, (task_losses, task_rewards, divergence) in enumerate(
        zip(metrics["task_losses"], metrics["task_rewards"], metrics["divergences"])
    ):
        print(f"\nTask {i + 1}:")
        print(f"  Average Loss: {sum(task_losses)/len(task_losses):.4f}")
        print(f"  Average Reward: {sum(task_rewards)/len(task_rewards):.2f}")
        print(f"  Parameter Divergence: {divergence:.4f}")
    
    print("\n" + "=" * 60)
    print("Key Features Demonstrated:")
    print("=" * 60)
    print("✓ Cross-modal model: Processes multi-modal inputs")
    print("✓ Internal/External signals: Recurrent feedback architecture")
    print("✓ Environmental interaction: Outputs feed back as inputs")
    print("✓ Continual learning: Dynamic adaptation across tasks")
    print("✓ Knowledge preservation: EWC prevents catastrophic forgetting")
    print("✓ Divergence monitoring: Tracks parameter changes")
    print("=" * 60)

    # Save a static summary visualization
    task_avg_losses = [sum(task_losses) / len(task_losses) for task_losses in metrics["task_losses"]]
    task_avg_rewards = [sum(task_rewards) / len(task_rewards) for task_rewards in metrics["task_rewards"]]
    divergences = metrics["divergences"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle("Continual Learning Summary", fontsize=14, fontweight="bold")

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

    axes[2].plot(divergences, marker="o", color="purple")
    axes[2].set_title("Parameter Divergence")
    axes[2].set_xlabel("Task")
    axes[2].set_ylabel("Divergence")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    output_path = "continual_learning_summary.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSummary visualization saved to {output_path}")


if __name__ == "__main__":
    main()
