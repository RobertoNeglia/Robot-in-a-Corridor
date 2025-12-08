"""
Train and visualize a single Q-Learning agent.
"""

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend to avoid display issues

from src.environment import Corridor
from src.agent import QLearningAgent
from src.training import train_td_learning
from src.visualization import generate_episode_gif
import matplotlib.pyplot as plt

print("Training Q-Learning agent on 100-unit corridor...")
print("=" * 60)

# Create environment and agent
corridor = Corridor(length=100)
agent = QLearningAgent(corridor, gamma=0.9, alpha=0.5, epsilon=0.3)

# Train
print("\nTraining for 2000 episodes...")
steps_history, reward_history = train_td_learning(
    corridor, agent, n_epochs=2000, max_steps=500
)

print(f"\nTraining complete!")
print(f"Final 100-episode average reward: {sum(reward_history[-100:])/100:.2f}")
print(f"Final 100-episode average steps: {sum(steps_history[-100:])/100:.2f}")

# Save training plots
print("\nSaving training plots to 'images/training_results.png'...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(reward_history)
ax1.set_xlabel("Episode")
ax1.set_ylabel("Total Reward")
ax1.set_title("Reward per Episode")
ax1.grid(True)

ax2.plot(steps_history)
ax2.set_xlabel("Episode")
ax2.set_ylabel("Steps")
ax2.set_title("Steps per Episode")
ax2.grid(True)

plt.tight_layout()
plt.savefig("images/training_results.png", dpi=150)
plt.close()
print("✓ Training plots saved")

# Generate GIF of trained agent
print("\nGenerating GIF of trained agent navigating the corridor...")
num_frames = generate_episode_gif(
    corridor=corridor,
    agent=agent,
    output_path="images/q_learning_trained.gif",
    show_frames=False,
    frame_interval=100,
    dpi=150,
    max_frames=200,
)

print(f"\n✓ GIF saved to 'images/q_learning_trained.gif' ({num_frames} frames)")
print("=" * 60)
