# Robot in a Corridor

A reinforcement learning project where a robot learns to navigate through a corridor to reach a goal. The project implements and compares multiple RL algorithms.

<img src="images/working-robot.gif">

## Features

- **Multiple RL Algorithms Implemented:**
  - **Monte Carlo** - Episode-based value function learning
  - **Q-Learning** - Off-policy TD control
  - **SARSA** - On-policy TD control
  - **Expected SARSA** - Stable on-policy learning with expected values

- **Modular Architecture:**
  - Clean separation of environment, agents, visualization, and training
  - Easy to extend with new algorithms or environments
  - Reusable components for experimentation

- **Comparison Tools:**
  - Built-in utilities to compare algorithm performance
  - Statistical analysis over multiple runs
  - Visualization of learning curves and confidence intervals

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Training a Single Agent

```python
from src.environment import Corridor
from src.agent import QLearningAgent
from src.training import train_td_learning
from src.visualization import plot_training_results

# Create environment and agent
corridor = Corridor(length=100)
agent = QLearningAgent(corridor, gamma=0.9, alpha=0.1, epsilon=0.5)

# Train
steps_history, reward_history = train_td_learning(
    corridor, agent, n_epochs=1000, max_steps=500
)

# Visualize results
plot_training_results(reward_history, steps_history)
```

### Comparing Multiple Algorithms

```python
from src.training import compare_algorithms
from src.agent import QLearningAgent, SARSAAgent, ExpectedSARSAAgent, MonteCarloAgent
from src.visualization import plot_algorithm_comparison

# Define algorithms to compare
algorithms = {
    'Q-Learning': QLearningAgent,
    'SARSA': SARSAAgent,
    'Expected SARSA': ExpectedSARSAAgent,
    'Monte Carlo': MonteCarloAgent,
}

# Run comparison
results = compare_algorithms(
    env_config={'length': 100, 'width': 5},
    algorithms=algorithms,
    n_epochs=1000,
    n_runs=5
)

# Visualize
plot_algorithm_comparison(results, metric='rewards', window=50)
```

## Project Structure

```
Robot-in-a-Corridor/
├── src/
│   ├── __init__.py          # Package exports
│   ├── utils.py             # Constants and utilities
│   ├── environment.py       # Corridor environment
│   ├── agent.py             # RL agent implementations
│   ├── training.py          # Training utilities
│   └── visualization.py     # Plotting and GIF generation
├── RL.ipynb                 # Original notebook (refactored)
├── examples.ipynb           # Algorithm comparison examples
├── requirements.txt         # Dependencies
└── README.md
```

## Algorithms Overview

### Monte Carlo
- **Type:** Episode-based learning
- **Update:** After complete episodes
- **Pros:** Simple, unbiased estimates
- **Cons:** High variance, requires episode completion

### Q-Learning
- **Type:** Off-policy TD control
- **Update:** After each step (bootstrapping)
- **Pros:** Fast learning, doesn't need complete episodes
- **Cons:** Can be unstable with high learning rates

### SARSA
- **Type:** On-policy TD control
- **Update:** After each step using actual next action
- **Pros:** More conservative, learns safe policies
- **Cons:** Slower convergence than Q-Learning

### Expected SARSA
- **Type:** Hybrid on-policy TD control
- **Update:** Uses expected value over next actions
- **Pros:** More stable than SARSA, better performance
- **Cons:** Slightly more computation per update

## Extending the Project

### Adding a New Algorithm

1. Inherit from `BaseAgent` in `src/agent.py`:

```python
class MyNewAgent(BaseAgent):
    def __init__(self, env, gamma=0.9, alpha=0.1, epsilon=0.5):
        super().__init__(env, gamma, alpha, epsilon)
        # Initialize your data structures
        
    def greedy_action(self, state, allowed_actions):
        # Implement greedy action selection
        pass
        
    def update(self, state, action, reward, next_state):
        # Implement your learning update
        pass
```

2. Export it in `src/__init__.py`
3. Use it with the existing training utilities

### Creating a New Environment

Implement the same interface as `Corridor`:
- `get_state()` - Return current state
- `get_allowed_actions(state)` - Return valid actions
- `update(action)` - Apply action and update environment
- `give_reward()` - Return reward for current state
- `is_terminal()` - Check if episode is done
- `reset_env()` - Reset to initial state

## Dependencies

- numpy - Numerical computations
- matplotlib - Visualization
- pillow - GIF generation
- ipython - Notebook support