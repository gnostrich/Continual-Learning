#!/usr/bin/env python3
"""
Example usage scenarios for the continual learning demonstration.
Run this script to see different network architectures and configurations in action.
"""
import subprocess
import sys


def run_example(description, command):
    """Run an example and print results."""
    print("\n" + "="*70)
    print(f"EXAMPLE: {description}")
    print("="*70)
    print(f"Command: {command}\n")
    
    result = subprocess.run(command, shell=True, capture_output=False)
    
    if result.returncode == 0:
        print(f"\n✓ {description} completed successfully!")
    else:
        print(f"\n✗ {description} failed with exit code {result.returncode}")
        sys.exit(1)


def main():
    """Run a series of example demonstrations."""
    print("\n" + "="*70)
    print("CONTINUAL LEARNING DEMONSTRATION - EXAMPLES")
    print("="*70)
    print("\nThis script will run several examples to demonstrate the system:\n")
    print("1. Feed-forward network with default settings")
    print("2. GRU network with default settings")
    print("3. Feed-forward network with custom learning rate")
    print("\nPress Ctrl+C at any time to stop.\n")
    
    input("Press Enter to continue...")
    
    # Example 1: Feed-forward network
    run_example(
        "Feed-forward network (30 episodes)",
        "python demo.py --network feedforward --episodes 30"
    )
    
    # Example 2: GRU network
    run_example(
        "GRU recurrent network (30 episodes)",
        "python demo.py --network gru --episodes 30"
    )
    
    # Example 3: Custom learning rate
    run_example(
        "Feed-forward with lower learning rate (30 episodes)",
        "python demo.py --network feedforward --episodes 30 --lr 0.005"
    )
    
    print("\n" + "="*70)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nCheck 'continual_learning_results.png' for the latest visualization.")
    print("\nKey takeaways:")
    print("- Both feed-forward and GRU networks can learn continually")
    print("- The system updates online (after each step)")
    print("- Visualizations show learning dynamics in real-time")
    print("- Outputs directly affect the environment, creating a feedback loop")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user.")
        sys.exit(0)
