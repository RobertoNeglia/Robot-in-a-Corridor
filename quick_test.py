"""
Quick test of Q-Learning agent.
"""

from src.environment import Corridor
from src.agent import QLearningAgent
from src.training import train_td_learning

print("Quick Q-Learning test...")

# Create environment and agent
corridor = Corridor(length=50)  # Shorter corridor for faster training
agent = QLearningAgent(corridor, gamma=0.9, alpha=0.5, epsilon=0.3)

# Train for fewer episodes
print("Training for 500 episodes...")
steps_history, reward_history = train_td_learning(
    corridor, agent, n_epochs=500, max_steps=200
)

# Show final performance
print(f"\nFinal 50-episode average reward: {sum(reward_history[-50:])/50:.2f}")
print(f"Final 50-episode average steps: {sum(steps_history[-50:])/50:.2f}")

# Test the trained agent
print("\nTesting trained agent...")
corridor.reset_env()
total_steps = 0
states_visited = []

while not corridor.is_terminal() and total_steps < 200:
    state = corridor.get_state()
    states_visited.append(state)
    allowed_actions = corridor.get_allowed_actions(state)
    action = agent.greedy_action(state, allowed_actions)
    corridor.update(action)
    total_steps += 1

if corridor.is_terminal():
    print(f"✓ Success! Agent reached the goal in {total_steps} steps")
else:
    print(f"✗ Agent did not reach goal in {total_steps} steps")

print(f"Final position: x={states_visited[-1][1]}, corridor length={corridor.length-1}")
