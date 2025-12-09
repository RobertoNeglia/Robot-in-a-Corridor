"""
Visualize how an agent's behavior improves during training.
This creates a GIF showing the agent at different training stages.
"""

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend

from src.agent import QLearningAgent
from src.visualization import visualize_training_progress

print("=" * 70)
print("MULTI-EPISODE TRAINING VISUALIZATION")
print("=" * 70)
print("\nThis will train a Q-Learning agent and capture snapshots")
print("of its behavior at different stages of training.")
print()

# Configuration
env_config = {"length": 50, "width": 5}  # Shorter corridor for faster visualization
agent_params = {"gamma": 0.9, "alpha": 0.5, "epsilon": 0.5}
episodes_to_show = [0, 50, 200, 500, 999]  # Episodes to capture
n_epochs = 1000

print(
    f"Environment: Corridor (length={env_config['length']}, width={env_config['width']})"
)
print(
    f"Agent: Q-Learning (gamma={agent_params['gamma']}, alpha={agent_params['alpha']}, epsilon={agent_params['epsilon']})"
)
print(f"Training episodes: {n_epochs}")
print(f"Snapshots at episodes: {episodes_to_show}")
print()

# Run visualization
results = visualize_training_progress(
    env_config=env_config,
    AgentClass=QLearningAgent,
    agent_params=agent_params,
    episodes_to_show=episodes_to_show,
    n_epochs=n_epochs,
    max_steps=200,
    output_path="images/training_evolution.gif",
)

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

# Show improvement over training
for episode in episodes_to_show:
    snapshot = results["snapshots"][episode]
    status = "✓ Reached goal" if snapshot["reached_goal"] else "✗ Did not reach goal"
    print(
        f"Episode {episode:4d}: {snapshot['steps']:3d} steps | {status} | "
        f"ε={snapshot['epsilon']:.3f} | Avg reward: {snapshot['avg_reward_last_50']:6.1f}"
    )

print("\n✓ Visualization complete!")
print("  - Training evolution GIF: images/training_evolution.gif")
print("\nOpen the GIF to see how the agent's behavior improves during training!")
print("=" * 70)
