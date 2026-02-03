"""
Simple test script to validate the continual learning demonstration components.
"""
import torch
import numpy as np
from networks import FeedForwardNetwork, GRUNetwork
from visualization import ContinualLearningVisualizer


def test_feedforward_network():
    """Test feed-forward network initialization and forward pass."""
    print("Testing Feed-forward Network...")
    
    input_size, hidden_size, output_size = 4, 32, 2
    network = FeedForwardNetwork(input_size, hidden_size, output_size)
    
    # Test forward pass
    test_input = torch.randn(input_size)
    output = network(test_input)
    
    assert output.shape == (output_size,), f"Expected shape {(output_size,)}, got {output.shape}"
    print("✓ Feed-forward network test passed!")
    return True


def test_gru_network():
    """Test GRU network initialization and forward pass."""
    print("Testing GRU Network...")
    
    input_size, hidden_size, output_size = 4, 32, 2
    network = GRUNetwork(input_size, hidden_size, output_size)
    
    # Test forward pass
    test_input = torch.randn(input_size)
    hidden = network.init_hidden()
    output, new_hidden = network(test_input, hidden)
    
    assert output.shape[1] == output_size, f"Expected output dim {output_size}, got {output.shape}"
    assert new_hidden.shape == hidden.shape, f"Hidden state shape mismatch"
    print("✓ GRU network test passed!")
    return True


def test_visualizer():
    """Test visualizer initialization and update."""
    print("Testing Visualizer...")
    
    viz = ContinualLearningVisualizer(max_history=100)
    
    # Add some test data
    for i in range(50):
        loss = np.random.random()
        reward = i * 0.5
        action = np.random.randint(0, 2)
        prediction = np.random.random((2,))
        viz.update(loss, reward, action, prediction)
    
    assert len(viz.losses) == 50, f"Expected 50 losses, got {len(viz.losses)}"
    assert len(viz.rewards) == 50, f"Expected 50 rewards, got {len(viz.rewards)}"
    
    viz.close()
    print("✓ Visualizer test passed!")
    return True


def test_network_training():
    """Test that networks can be trained with backpropagation."""
    print("Testing Network Training...")
    
    input_size, hidden_size, output_size = 4, 32, 2
    
    # Test feed-forward
    network = FeedForwardNetwork(input_size, hidden_size, output_size)
    optimizer = torch.optim.Adam(network.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    
    # Training step
    test_input = torch.randn(input_size)
    target = torch.randn(output_size)
    output = network(test_input)
    loss = criterion(output, target)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print("✓ Network training test passed!")
    return True


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("CONTINUAL LEARNING DEMONSTRATION - COMPONENT TESTS")
    print("="*60 + "\n")
    
    tests = [
        test_feedforward_network,
        test_gru_network,
        test_visualizer,
        test_network_training
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test failed with error: {e}")
            failed += 1
        print()
    
    print("="*60)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("="*60)
    
    if failed == 0:
        print("\n✓ All tests passed successfully!")
        return 0
    else:
        print(f"\n✗ {failed} test(s) failed")
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
