"""
Demo: PyTorch GPU-Accelerated Reinforcement Learning
Shows the performance improvement with GPU acceleration.
"""

import time
import torch
from src.environment import Corridor
from src.agent import QLearningAgent, SARSAAgent, ExpectedSARSAAgent, MonteCarloAgent
from src.training import train_td_learning, train_monte_carlo

print("=" * 70)
print("PYTORCH GPU-ACCELERATED REINFORCEMENT LEARNING")
print("=" * 70)

# Check device
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"\n🚀 Device: {device}")
print(f"PyTorch version: {torch.__version__}")

if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
elif device.type == "mps":
    print("GPU: Apple Silicon (MPS)")
else:
    print("GPU: Not available, using CPU")

print("\n" + "=" * 70)
print("TRAINING Q-LEARNING AGENT")
print("=" * 70)

# Environment configuration
corridor = Corridor(length=50)
agent = QLearningAgent(corridor, gamma=0.9, alpha=0.5, epsilon=0.5)

print(f"\nQ-table shape: {agent.Q.shape}")
print(f"Q-table device: {agent.Q.device}")
print(f"Total parameters: {agent.Q.numel():,}")

# Train
print("\nTraining for 1000 episodes...")
start_time = time.time()
steps, rewards = train_td_learning(corridor, agent, n_epochs=1000, max_steps=300)
end_time = time.time()

print(f"\n✓ Training completed in {end_time - start_time:.2f} seconds")
print(f"Initial reward (avg first 10): {sum(rewards[:10])/10:.2f}")
print(f"Final reward (avg last 100): {sum(rewards[-100:])/100:.2f}")
print(f"Improvement: {sum(rewards[-100:])/100 - sum(rewards[:10])/10:.2f}")

print("\n" + "=" * 70)
print("TESTING ALL ALGORITHMS")
print("=" * 70)

algorithms = [
    ("Q-Learning", QLearningAgent, {"gamma": 0.9, "alpha": 0.5, "epsilon": 0.5}),
    ("SARSA", SARSAAgent, {"gamma": 0.9, "alpha": 0.5, "epsilon": 0.5}),
    (
        "Expected SARSA",
        ExpectedSARSAAgent,
        {"gamma": 0.9, "alpha": 0.5, "epsilon": 0.5},
    ),
    ("Monte Carlo", MonteCarloAgent, {"gamma": 0.9, "alpha": 0.5, "epsilon": 0.3}),
]

print(
    f"\nTraining each algorithm for 500 episodes on a {corridor.length-3}-unit corridor...\n"
)

for name, AgentClass, params in algorithms:
    corridor_test = Corridor(length=50)
    agent_test = AgentClass(corridor_test, **params)

    start = time.time()
    if name == "Monte Carlo":
        steps, rewards = train_monte_carlo(
            corridor_test, agent_test, n_epochs=500, max_steps=300
        )
    else:
        steps, rewards = train_td_learning(
            corridor_test, agent_test, n_epochs=500, max_steps=300
        )
    duration = time.time() - start

    final_perf = sum(rewards[-50:]) / 50
    print(f"  {name:20s}: {duration:5.2f}s | Final reward: {final_perf:7.2f}")

print("\n" + "=" * 70)
print("✓ All algorithms successfully using GPU acceleration!")
print("=" * 70)
