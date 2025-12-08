"""
Quick comparison of all implemented RL algorithms.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.environment import Corridor
from src.agent import MonteCarloAgent, QLearningAgent, SARSAAgent, ExpectedSARSAAgent
from src.training import compare_algorithms
from src.visualization import plot_algorithm_comparison

print("=" * 70)
print("REINFORCEMENT LEARNING ALGORITHM COMPARISON")
print("=" * 70)
print("\nComparing 4 algorithms:")
print("  1. Q-Learning (Off-policy TD)")
print("  2. SARSA (On-policy TD)")
print("  3. Expected SARSA (Stable on-policy)")
print("  4. Monte Carlo (Episode-based)")
print("\nEnvironment: Corridor (length=100, width=5)")
print("Training: 2000 episodes, 3 independent runs per algorithm")
print("=" * 70)

# Define algorithms to compare
# Note: Using higher learning rate (alpha=0.5) and lower initial epsilon
algorithms = {
    "Q-Learning": lambda env: QLearningAgent(env, gamma=0.9, alpha=0.5, epsilon=0.3),
    "SARSA": lambda env: SARSAAgent(env, gamma=0.9, alpha=0.5, epsilon=0.3),
    "Expected SARSA": lambda env: ExpectedSARSAAgent(
        env, gamma=0.9, alpha=0.5, epsilon=0.3
    ),
    "Monte Carlo": lambda env: MonteCarloAgent(env, gamma=0.9, alpha=0.5, epsilon=0.3),
}

# Run comparison
print("\nStarting comparison... This may take a few minutes.\n")
results = compare_algorithms(
    env_config={"length": 100, "width": 5},
    algorithms=algorithms,
    n_epochs=2000,
    max_steps=500,
    n_runs=3,
)

print("\n" + "=" * 70)
print("COMPARISON COMPLETE!")
print("=" * 70)

# Plot reward comparison
print("\nGenerating reward comparison plot...")
plot_algorithm_comparison(results, metric="rewards", window=20)

# Plot steps comparison
print("\nGenerating steps comparison plot...")
plot_algorithm_comparison(results, metric="steps", window=20)

print("\n" + "=" * 70)
print("All done! Check the plots above for detailed comparison.")
print("=" * 70)
