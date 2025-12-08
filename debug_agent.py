"""
Debug the environment and agent behavior.
"""

from src.environment import Corridor
from src.agent import QLearningAgent
from src.utils import ACTIONS

# Create a small corridor for debugging
corridor = Corridor(length=10, width=5)
print(f"Corridor dimensions: {corridor.corridor.shape}")
print(f"Corridor length (with walls): {corridor.length}")
print(f"Start position: {corridor.agent_pos}")
print(f"Goal at x={corridor.length-1}")

# Get initial state
state = corridor.get_state()
print(f"\nInitial state: y={state[0]}, x={state[1]}, orientation={state[2]}")

# Check allowed actions
allowed = corridor.get_allowed_actions(state)
print(f"Allowed actions: {[a.name for a in allowed]}")

# Test moving forward
print("\n--- Testing FORWARD action ---")
if ACTIONS.FORWARD in allowed:
    corridor.update(ACTIONS.FORWARD)
    new_state = corridor.get_state()
    reward = corridor.give_reward()
    print(
        f"After FORWARD: y={new_state[0]}, x={new_state[1]}, orientation={new_state[2]}"
    )
    print(f"Reward: {reward}")
    print(f"Is terminal: {corridor.is_terminal()}")

    # Try a few more steps
    for i in range(5):
        state = corridor.get_state()
        if ACTIONS.FORWARD in corridor.get_allowed_actions(state):
            corridor.update(ACTIONS.FORWARD)
            new_state = corridor.get_state()
            print(f"Step {i+2}: x={new_state[1]}, reward={corridor.give_reward()}")
        else:
            print(f"Step {i+2}: Cannot move forward")
            break

# Test with agent
print("\n--- Testing with Q-Learning Agent ---")
corridor.reset_env()
agent = QLearningAgent(corridor, gamma=0.9, alpha=0.5, epsilon=0.0)  # No exploration

print(f"Initial Q-values sample:")
state = corridor.get_state()
for action in corridor.get_allowed_actions(state)[:3]:
    q_val = agent.Q.get((state, action), agent.default_q)
    print(f"  Q({state}, {action.name}) = {q_val}")

# Train for a few episodes
print("\nTraining for 200 episodes with epsilon=0.5...")
agent.epsilon = 0.5  # More exploration
for ep in range(200):
    corridor.reset_env()
    steps = 0
    total_reward = 0

    while not corridor.is_terminal() and steps < 100:  # Increased max steps
        state = corridor.get_state()
        action = agent.choose_action(state)
        corridor.update(action)
        next_state, reward = corridor.get_state_and_reward()
        agent.update(state, action, reward, next_state)
        total_reward += reward
        steps += 1

    agent.reset_episode()
    agent.decay_epsilon()

    if ep % 40 == 0:
        print(
            f"Episode {ep}: steps={steps}, reward={total_reward:.0f}, reached_goal={corridor.is_terminal()}, epsilon={agent.epsilon:.3f}"
        )

# Test final policy
print("\n--- Testing Learned Policy ---")
corridor.reset_env()
agent.epsilon = 0.0  # Greedy only
path = []

for step in range(20):
    state = corridor.get_state()
    path.append((state[1], state[0]))  # (x, y)

    if corridor.is_terminal():
        print(f"✓ Reached goal in {step} steps!")
        break

    allowed = corridor.get_allowed_actions(state)
    action = agent.greedy_action(state, allowed)
    corridor.update(action)
    print(f"Step {step}: x={state[1]}, action={action.name}")
else:
    print(f"✗ Did not reach goal in 20 steps")

print(f"\nPath taken (x positions): {[p[0] for p in path]}")
