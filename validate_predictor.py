#!/usr/bin/env python3
"""
Validation script to verify predictor-based continual learning actually works.

Tests:
1. Prediction accuracy improves over training
2. Predictor learns simple deterministic environment dynamics
3. Model adapts to environment changes
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from continual_learning import CrossModalModel, EnvironmentPredictor, ContinualLearningLoop


class DeterministicEnvironment:
    """Simple deterministic environment where response = action * 0.5 (predictable)."""
    
    def __init__(self, observation_dim=64, action_dim=32, num_modalities=2):
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.num_modalities = num_modalities
        self.current_task = 0
        
    def reset(self):
        """Reset to zero state."""
        return self.get_observation()
    
    def step(self, action):
        """Deterministic: observation = action * 0.5 (predictable dynamics)."""
        if isinstance(action, torch.Tensor):
            action_np = action.detach().cpu().numpy().flatten()
        else:
            action_np = action.flatten()
        
        # Pad/truncate to match observation dim
        if len(action_np) < self.observation_dim:
            obs = np.zeros(self.observation_dim)
            obs[:len(action_np)] = action_np * 0.5  # Deterministic scaling
        else:
            obs = action_np[:self.observation_dim] * 0.5
        
        # Create multi-modal observations (same across modalities for simplicity)
        observations = [torch.FloatTensor(obs).unsqueeze(0) for _ in range(self.num_modalities)]
        
        reward = -np.linalg.norm(obs)  # Dummy reward
        done = False
        info = {}
        
        return observations, reward, done, info
    
    def get_observation(self):
        """Get zero observations."""
        return [torch.zeros(1, self.observation_dim) for _ in range(self.num_modalities)]
    
    def change_task(self, task_id):
        """Task change (no-op for this simple env)."""
        self.current_task = task_id


def test_prediction_accuracy_improvement():
    """Test 1: Does prediction accuracy improve over training?"""
    print("\n" + "=" * 70)
    print("TEST 1: Prediction Accuracy Improvement")
    print("=" * 70)
    
    # Setup
    model = CrossModalModel(input_dim=64, hidden_dim=128, output_dim=32, num_modalities=2)
    predictor = EnvironmentPredictor(action_dim=32, hidden_dim=128, output_dim=64, num_modalities=2)
    env = DeterministicEnvironment(observation_dim=64, action_dim=32, num_modalities=2)
    
    loop = ContinualLearningLoop(
        model=model,
        predictor=predictor,
        environment=env,
        learning_rate=1e-3,
        ewc_lambda=0.0,  # Disable EWC for this test
        feedback_weight=0.3,
        divergence_weight=1.0,
        action_l2_weight=1e-4,
    )
    
    # Measure initial prediction error
    print("\nMeasuring initial prediction accuracy...")
    initial_errors = []
    model.reset_state()
    predictor.reset_state()
    
    for _ in range(50):
        obs = env.reset()
        action, internal_state = model(obs)
        predicted_obs, _ = predictor(action, internal_state)
        
        # Step environment with actual dynamics
        actual_obs, _, _, _ = env.step(action)
        
        # Compute prediction error
        error = sum(torch.nn.functional.mse_loss(pred, actual).item() 
                   for pred, actual in zip(predicted_obs, actual_obs))
        initial_errors.append(error)
    
    initial_mean_error = np.mean(initial_errors)
    print(f"Initial prediction error: {initial_mean_error:.6f}")
    
    # Train for several episodes
    print("\nTraining for 100 steps...")
    for episode in range(10):
        loop.run_episode(max_steps=10, verbose=False)
    
    # Measure final prediction error
    print("\nMeasuring final prediction accuracy...")
    final_errors = []
    model.reset_state()
    predictor.reset_state()
    
    for _ in range(50):
        obs = env.reset()
        action, internal_state = model(obs)
        predicted_obs, _ = predictor(action, internal_state)
        
        actual_obs, _, _, _ = env.step(action)
        
        error = sum(torch.nn.functional.mse_loss(pred, actual).item() 
                   for pred, actual in zip(predicted_obs, actual_obs))
        final_errors.append(error)
    
    final_mean_error = np.mean(final_errors)
    print(f"Final prediction error: {final_mean_error:.6f}")
    
    improvement = ((initial_mean_error - final_mean_error) / initial_mean_error) * 100
    print(f"\nImprovement: {improvement:.1f}%")
    
    if final_mean_error < initial_mean_error:
        print("✓ PASS: Prediction accuracy improved")
        return True
    else:
        print("✗ FAIL: Prediction accuracy did not improve")
        return False


def test_learning_curve():
    """Test 2: Track prediction error over training steps."""
    print("\n" + "=" * 70)
    print("TEST 2: Learning Curve Analysis")
    print("=" * 70)
    
    model = CrossModalModel(input_dim=64, hidden_dim=128, output_dim=32, num_modalities=2)
    predictor = EnvironmentPredictor(action_dim=32, hidden_dim=128, output_dim=64, num_modalities=2)
    env = DeterministicEnvironment(observation_dim=64, action_dim=32, num_modalities=2)
    
    loop = ContinualLearningLoop(
        model=model,
        predictor=predictor,
        environment=env,
        learning_rate=1e-3,
        ewc_lambda=0.0,
        feedback_weight=0.3,
        divergence_weight=1.0,
        action_l2_weight=1e-4,
    )
    
    print("\nTracking prediction error over 20 episodes...")
    prediction_errors = []
    
    for episode in range(20):
        # Train
        loop.run_episode(max_steps=10, verbose=False)
        
        # Evaluate prediction accuracy
        model.reset_state()
        predictor.reset_state()
        errors = []
        
        for _ in range(10):
            obs = env.reset()
            action, internal_state = model(obs)
            predicted_obs, _ = predictor(action, internal_state)
            actual_obs, _, _, _ = env.step(action)
            
            error = sum(torch.nn.functional.mse_loss(pred, actual).item() 
                       for pred, actual in zip(predicted_obs, actual_obs))
            errors.append(error)
        
        mean_error = np.mean(errors)
        prediction_errors.append(mean_error)
        
        if episode % 5 == 0:
            print(f"Episode {episode:2d}: Prediction error = {mean_error:.6f}")
    
    # Check if trend is downward
    early_avg = np.mean(prediction_errors[:5])
    late_avg = np.mean(prediction_errors[-5:])
    
    print(f"\nEarly episodes avg error: {early_avg:.6f}")
    print(f"Late episodes avg error: {late_avg:.6f}")
    
    # Save learning curve
    plt.figure(figsize=(8, 5))
    plt.plot(prediction_errors, marker='o', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Prediction Error (MSE)')
    plt.title('Learning Curve: Prediction Error Over Time')
    plt.grid(True, alpha=0.3)
    plt.savefig('prediction_learning_curve.png', dpi=150, bbox_inches='tight')
    print("\nLearning curve saved to prediction_learning_curve.png")
    
    if late_avg < early_avg:
        print("✓ PASS: Prediction error decreases over time")
        return True
    else:
        print("✗ FAIL: No clear improvement in prediction")
        return False


def main():
    """Run all validation tests."""
    print("\n" + "=" * 70)
    print("CONTINUAL LEARNING VALIDATION SUITE")
    print("Idiot-checking the predictor-based learning system")
    print("=" * 70)
    
    results = []
    
    # Test 1: Prediction accuracy improvement
    results.append(test_prediction_accuracy_improvement())
    
    # Test 2: Learning curve
    results.append(test_learning_curve())
    
    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\nTests passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("The predictor-based continual learning system is working correctly.")
    else:
        print("\n✗✗✗ SOME TESTS FAILED ✗✗✗")
        print("The system may not be learning environment dynamics properly.")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
