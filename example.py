"""Example usage of the continual learning system."""

import torch
from continual_learning import CrossModalModel, Environment, ContinualLearningLoop


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
    
    # 2. Simulated environment
    environment = Environment(
        observation_dim=input_dim,
        action_dim=output_dim,
        num_modalities=num_modalities,
        task_difficulty=0.5,
    )
    print(f"\n✓ Environment initialized")
    print(f"  Observation dim: {environment.observation_dim}")
    print(f"  Action dim: {environment.action_dim}")
    
    # 3. Continual learning loop
    learning_loop = ContinualLearningLoop(
        model=model,
        environment=environment,
        learning_rate=1e-3,
        ewc_lambda=0.5,  # Regularization strength for knowledge preservation
        feedback_weight=0.3,  # Weight for recurrent feedback
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


if __name__ == "__main__":
    main()
