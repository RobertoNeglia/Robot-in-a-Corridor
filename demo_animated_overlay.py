"""
Demo: Animated Multi-Agent Overlay Visualization
Shows agents from different training stages appearing progressively in an animated GIF.
"""

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend

from src import QLearningAgent, visualize_training_progress_animated_overlay

# Environment configuration
env_config = {
    "length": 23,  # Short corridor for clearer visualization
    "width": 5,
}

# Agent parameters
agent_params = {
    "gamma": 0.9,
    "alpha": 0.5,
    "epsilon": 0.3,
}

# Episodes to capture and overlay
episodes_to_show = [0, 50, 100, 200, 400, 499]

print("\n" + "=" * 70)
print("ANIMATED MULTI-AGENT OVERLAY DEMO")
print("=" * 70)
print(f"\nThis will create an animated GIF showing {len(episodes_to_show)} agents")
print("appearing progressively, each representing a different training stage.")
print(f"\nEpisodes to visualize: {episodes_to_show}")
print("\nThe animation will show each agent being added one by one,")
print("revealing how the agent's behavior evolves during training.")
print("=" * 70)

# Run visualization
results = visualize_training_progress_animated_overlay(
    env_config=env_config,
    AgentClass=QLearningAgent,
    agent_params=agent_params,
    episodes_to_show=episodes_to_show,
    n_epochs=500,
    max_steps=200,
    output_path="images/training_animated_overlay.gif",
)

print("\n" + "=" * 70)
print("TRAINING SUMMARY")
print("=" * 70)
print("\nAgent performance at captured episodes:")
for episode in episodes_to_show:
    snapshot = results["snapshots"][episode]
    goal_status = "✓" if snapshot["reached_goal"] else "✗"
    print(
        f"  Episode {episode:3d}: {snapshot['steps']:2d} steps | {goal_status} | "
        f"Avg Reward: {snapshot['avg_reward']:6.1f}"
    )

print("\n" + "=" * 70)
print("✓ Demo complete! Check 'images/training_animated_overlay.gif'")
print("=" * 70)
