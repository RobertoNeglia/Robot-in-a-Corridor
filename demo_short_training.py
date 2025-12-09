"""
Quick demo showing agent learning with a shorter corridor (easier to learn).
"""

import matplotlib

matplotlib.use("Agg")

from src.agent import QLearningAgent
from src.visualization import visualize_training_progress

print("Training Q-Learning on short corridor (length=20)...")
print("This should show clear improvement from random to optimal behavior.\n")

# Shorter corridor - easier to learn and visualize
results = visualize_training_progress(
    env_config={"length": 20, "width": 5},
    AgentClass=QLearningAgent,
    agent_params={"gamma": 0.9, "alpha": 0.5, "epsilon": 0.5},
    episodes_to_show=[0, 100, 300, 499],  # Last episode is 499 (0-indexed)
    n_epochs=500,
    max_steps=100,
    output_path="images/training_short_corridor.gif",
)

print("\n" + "=" * 60)
for episode in [0, 100, 300, 499]:
    if episode in results["snapshots"]:
        snap = results["snapshots"][episode]
        status = "✓" if snap["reached_goal"] else "✗"
        print(
            f"Episode {episode:3d}: {snap['steps']:2d} steps | {status} | "
            f"Reward: {snap['avg_reward_last_50']:5.1f}"
        )
print("=" * 60)
