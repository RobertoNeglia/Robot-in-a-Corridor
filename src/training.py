"""
Training utilities for reinforcement learning agents.
"""

import numpy as np
from .agent import MonteCarloAgent, QLearningAgent, SARSAAgent, ExpectedSARSAAgent
from .visualization import plot_training_progress


def train_monte_carlo(env, agent, n_epochs=1000, max_steps=500, display_interval=None):
    """
    Train an agent using Monte Carlo method.

    Args:
        env: Environment instance
        agent: MonteCarloAgent instance
        n_epochs: Number of training episodes
        max_steps: Maximum steps per episode
        display_interval: How often to display progress (None for no display)

    Returns:
        Tuple of (steps_history, reward_history)
    """
    steps_history = []
    reward_history = []

    for i in range(n_epochs):
        # Run an episode
        while not env.is_terminal():
            state = env.get_state()
            action = agent.choose_action(state)
            env.update(action)
            next_state, reward = env.get_state_and_reward()
            agent.update_trajectory(next_state, reward)

            if agent.steps > max_steps:
                break

        reward_history.append(agent.total_reward)
        steps_history.append(agent.steps)

        if display_interval and i % display_interval == 0:
            plot_training_progress(
                env, i, steps_history[-1], reward_history[-1], agent.epsilon
            )

        # Update value function and reset
        agent.update()

    return steps_history, reward_history


def train_td_learning(env, agent, n_epochs=1000, max_steps=500, display_interval=None):
    """
    Train an agent using TD learning methods (Q-Learning, SARSA, Expected SARSA).

    Args:
        env: Environment instance
        agent: QLearningAgent, SARSAAgent, or ExpectedSARSAAgent instance
        n_epochs: Number of training episodes
        max_steps: Maximum steps per episode
        display_interval: How often to display progress (None for no display)

    Returns:
        Tuple of (steps_history, reward_history)
    """
    steps_history = []
    reward_history = []

    for i in range(n_epochs):
        env.reset_env()
        agent.reset_episode()

        state = env.get_state()

        # For SARSA, we need to choose the first action
        if isinstance(agent, SARSAAgent):
            action = agent.choose_action(state)

        # Run episode
        while not env.is_terminal():
            if not isinstance(agent, SARSAAgent):
                # Q-Learning and Expected SARSA choose action here
                action = agent.choose_action(state)

            env.update(action)
            next_state, reward = env.get_state_and_reward()

            # Update based on algorithm type
            if isinstance(agent, SARSAAgent):
                # SARSA needs next action for update
                if not env.is_terminal():
                    next_action = agent.choose_action(next_state)
                else:
                    next_action = None
                agent.update(state, action, reward, next_state, next_action)
                action = next_action
            else:
                # Q-Learning and Expected SARSA
                agent.update(state, action, reward, next_state)

            state = next_state

            if agent.steps > max_steps:
                break

        reward_history.append(agent.total_reward)
        steps_history.append(agent.steps)

        if display_interval and i % display_interval == 0:
            plot_training_progress(
                env, i, steps_history[-1], reward_history[-1], agent.epsilon
            )

        # Decay exploration rate
        agent.decay_epsilon()

    return steps_history, reward_history


def compare_algorithms(
    env_config, algorithms, agent_config, n_epochs=1000, max_steps=500, n_runs=5
):
    """
    Compare multiple RL algorithms on the same environment.

    Args:
        env_config: Dictionary with environment configuration (e.g., {'length': 100, 'width': 5})
        algorithms: Dictionary mapping algorithm names to agent classes
        agent_config: Dictionary mapping algorithm names to their specific parameters
        n_epochs: Number of training episodes per run
        max_steps: Maximum steps per episode
        n_runs: Number of independent runs for each algorithm

    Returns:
        Dictionary mapping algorithm names to performance statistics
    """
    from .environment import Corridor

    results = {}

    for algo_name, AgentClass in algorithms.items():
        print(f"\nTraining {algo_name}...")

        all_rewards = []
        all_steps = []
        final_rewards = []

        for run in range(n_runs):
            # Create fresh environment and agent
            env = Corridor(**env_config)
            agent = AgentClass(env, **agent_config.get(algo_name, {}))

            # Train
            if isinstance(agent, MonteCarloAgent):
                steps, rewards = train_monte_carlo(env, agent, n_epochs, max_steps)
            else:
                steps, rewards = train_td_learning(env, agent, n_epochs, max_steps)

            all_rewards.append(rewards)
            all_steps.append(steps)
            final_rewards.append(
                np.mean(rewards[-100:])
            )  # Average of last 100 episodes

            print(f"  Run {run+1}/{n_runs}: Final avg reward = {final_rewards[-1]:.2f}")

        # Compute statistics
        results[algo_name] = {
            "all_rewards": all_rewards,
            "all_steps": all_steps,
            "mean_rewards": np.mean(all_rewards, axis=0),
            "std_rewards": np.std(all_rewards, axis=0),
            "mean_steps": np.mean(all_steps, axis=0),
            "std_steps": np.std(all_steps, axis=0),
            "final_performance_mean": np.mean(final_rewards),
            "final_performance_std": np.std(final_rewards),
        }

    return results
