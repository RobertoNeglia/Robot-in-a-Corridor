"""
Agent classes implementing various reinforcement learning algorithms with PyTorch GPU acceleration.
"""

import numpy as np
import torch
import torch.nn as nn
from .utils import ORIENTATION, ACTIONS, TAKE_A_STEP, MOVE


# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


# Create action to index mapping (ACTIONS enum starts from 1, not 0)
ACTION_TO_INDEX = {action: idx for idx, action in enumerate(ACTIONS)}
INDEX_TO_ACTION = {idx: action for action, idx in ACTION_TO_INDEX.items()}


def state_to_index(state, env):
    """
    Convert state tuple to unique index for tensor operations.
    
    Args:
        state: (y, x, orientation) tuple
        env: Environment instance
        
    Returns:
        Unique integer index for the state
    """
    y, x, orientation = state
    height = env.corridor.shape[0] - 2  # Exclude walls
    width = env.corridor.shape[1] - 1   # Exclude left wall
    
    # Map to indices starting from 0
    y_idx = y - 1
    x_idx = x - 1
    ori_idx = orientation.value
    
    # Flatten to single index
    return y_idx * (width * 4) + x_idx * 4 + ori_idx


def get_state_space_size(env):
    """Get total number of possible states."""
    height = env.corridor.shape[0] - 2  # Exclude walls
    width = env.corridor.shape[1] - 1   # Exclude left wall
    n_orientations = 4
    return height * width * n_orientations


class BaseAgent:
    """
    Base class for reinforcement learning agents with PyTorch support.
    """

    def __init__(self, env, gamma=0.9, alpha=0.1, epsilon=0.5):
        """
        Constructor of the base agent class.

        Args:
            env: Environment instance
            gamma: Discount factor
            alpha: Learning rate
            epsilon: Exploration rate (0: greedy, 1: random)
        """
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        self.total_reward = 0
        self.steps = 0
        self.device = device

    def choose_action(self, state):
        """
        Choose an action using epsilon-greedy policy.

        Args:
            state: Current state

        Returns:
            Action chosen
        """
        allowed_actions = self.env.get_allowed_actions(state)
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(allowed_actions)
        else:
            return self.greedy_action(state, allowed_actions)

    def greedy_action(self, state, allowed_actions):
        """
        Get the greedy action for a given state (to be overridden).

        Args:
            state: State to check
            allowed_actions: Allowed actions for the given state

        Returns:
            Best action for the given state
        """
        raise NotImplementedError("Must be implemented by subclass")

    def reset_episode(self):
        """
        Reset episode-specific tracking.
        """
        self.total_reward = 0
        self.steps = 0
        self.env.reset_env()

    def decay_epsilon(self, decay_rate=0.99):
        """
        Decay the exploration rate.

        Args:
            decay_rate: Rate at which epsilon decays
        """
        self.epsilon = self.epsilon * decay_rate


class MonteCarloAgent(BaseAgent):
    """
    Agent using Monte Carlo learning with state-value function (PyTorch GPU-accelerated).
    """

    def __init__(self, env, gamma=0.9, alpha=0.1, epsilon=0.5):
        """
        Constructor of the class.

        Args:
            env: Environment instance
            gamma: Discount factor
            alpha: Learning rate
            epsilon: Exploration rate (0: greedy, 1: random)
        """
        super().__init__(env, gamma, alpha, epsilon)
        
        # Value function as PyTorch tensor
        n_states = get_state_space_size(env)
        self.V = torch.rand(n_states, device=self.device, dtype=torch.float32) * 0.99 + 0.01
        
        # History
        self.trajectory = []

    def greedy_action(self, state, allowed_actions):
        """
        Get the best action for a given state based on value function.

        Args:
            state: State to check
            allowed_actions: Allowed actions for the given state

        Returns:
            Best action for the given state
        """
        best_action = None
        best_value = float('-inf')
        
        # Compute the value of each action and return the best one
        for action in allowed_actions:
            y, x, orientation = state
            if action == ACTIONS.TURN_LEFT:
                orientation = ORIENTATION(((orientation.value + 1) % 4))
            elif action == ACTIONS.TURN_RIGHT:
                orientation = ORIENTATION(((orientation.value - 1) % 4))
            else:
                y += TAKE_A_STEP[orientation][0] * MOVE[action.name]
                x += TAKE_A_STEP[orientation][1] * MOVE[action.name]
            
            next_state_idx = state_to_index((y, x, orientation), self.env)
            value = self.V[next_state_idx].item()
            
            if value > best_value:
                best_value = value
                best_action = action
                
        return best_action

    def update_trajectory(self, state, reward):
        """
        Update the trajectory of the agent.

        Args:
            state: Current state
            reward: Reward received
        """
        self.steps += 1
        self.total_reward += reward
        self.trajectory.append((state, reward))

    def update(self):
        """
        Update the value function using the Monte Carlo method with PyTorch.
        """
        g = 0
        states_visited = []
        returns = []
        
        # Calculate returns for each state
        for state, reward in reversed(self.trajectory):
            g = self.gamma * g + reward
            states_visited.append(state_to_index(state, self.env))
            returns.append(g)
        
        # Reverse to get chronological order
        states_visited.reverse()
        returns.reverse()
        
        # Batch update using PyTorch
        if states_visited:
            state_indices = torch.tensor(states_visited, device=self.device, dtype=torch.long)
            returns_tensor = torch.tensor(returns, device=self.device, dtype=torch.float32)
            
            # TD error
            td_errors = returns_tensor - self.V[state_indices]
            
            # Update values
            self.V[state_indices] += self.alpha * td_errors
        
        # Reset the history
        self.trajectory = []
        self.reset_episode()
        self.decay_epsilon()


class QLearningAgent(BaseAgent):
    """
    Agent using Q-Learning (off-policy TD control) with PyTorch GPU acceleration.
    """

    def __init__(self, env, gamma=0.9, alpha=0.1, epsilon=0.5):
        """
        Constructor of the class.

        Args:
            env: Environment instance
            gamma: Discount factor
            alpha: Learning rate
            epsilon: Exploration rate
        """
        super().__init__(env, gamma, alpha, epsilon)
        
        # Q-function as PyTorch tensor [n_states, n_actions]
        n_states = get_state_space_size(env)
        n_actions = len(ACTIONS)
        self.Q = torch.zeros(n_states, n_actions, device=self.device, dtype=torch.float32)
        self.default_q = 0.0

    def greedy_action(self, state, allowed_actions):
        """
        Get the greedy action with highest Q-value.

        Args:
            state: Current state
            allowed_actions: Allowed actions for the given state

        Returns:
            Action with highest Q-value
        """
        state_idx = state_to_index(state, self.env)
        
        # Get Q-values for allowed actions
        q_values = []
        for action in allowed_actions:
            action_idx = ACTION_TO_INDEX[action]
            q_val = self.Q[state_idx, action_idx].item()
            q_values.append(q_val)
        
        max_q = max(q_values)
        # Handle ties by random selection
        best_actions = [
            action for action, q in zip(allowed_actions, q_values) if q == max_q
        ]
        return np.random.choice(best_actions)

    def update(self, state, action, reward, next_state):
        """
        Update Q-function using Q-Learning update rule.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
        """
        self.steps += 1
        self.total_reward += reward

        state_idx = state_to_index(state, self.env)
        action_idx = ACTION_TO_INDEX[action]
        
        # Q-Learning: off-policy update using max Q-value of next state
        if self.env.is_terminal():
            max_next_q = 0.0
        else:
            next_state_idx = state_to_index(next_state, self.env)
            next_allowed = self.env.get_allowed_actions(next_state)
            next_action_indices = torch.tensor([ACTION_TO_INDEX[a] for a in next_allowed], 
                                              device=self.device, dtype=torch.long)
            max_next_q = self.Q[next_state_idx, next_action_indices].max().item()

        # Q-Learning update using PyTorch
        current_q = self.Q[state_idx, action_idx]
        target = reward + self.gamma * max_next_q
        self.Q[state_idx, action_idx] = current_q + self.alpha * (target - current_q)


class SARSAAgent(BaseAgent):
    """
    Agent using SARSA (on-policy TD control) with PyTorch GPU acceleration.
    """

    def __init__(self, env, gamma=0.9, alpha=0.1, epsilon=0.5):
        """
        Constructor of the class.

        Args:
            env: Environment instance
            gamma: Discount factor
            alpha: Learning rate
            epsilon: Exploration rate
        """
        super().__init__(env, gamma, alpha, epsilon)
        
        # Q-function as PyTorch tensor
        n_states = get_state_space_size(env)
        n_actions = len(ACTIONS)
        self.Q = torch.zeros(n_states, n_actions, device=self.device, dtype=torch.float32)
        self.default_q = 0.0

    def greedy_action(self, state, allowed_actions):
        """
        Get the greedy action with highest Q-value.

        Args:
            state: Current state
            allowed_actions: Allowed actions for the given state

        Returns:
            Action with highest Q-value
        """
        state_idx = state_to_index(state, self.env)
        
        q_values = []
        for action in allowed_actions:
            q_val = self.Q[state_idx, ACTION_TO_INDEX[action]].item()
            q_values.append(q_val)
        
        max_q = max(q_values)
        best_actions = [
            action for action, q in zip(allowed_actions, q_values) if q == max_q
        ]
        return np.random.choice(best_actions)

    def update(self, state, action, reward, next_state, next_action):
        """
        Update Q-function using SARSA update rule.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            next_action: Next action (from policy)
        """
        self.steps += 1
        self.total_reward += reward

        state_idx = state_to_index(state, self.env)
        action_idx = ACTION_TO_INDEX[action]
        
        # SARSA: on-policy update using Q-value of next state-action pair
        if self.env.is_terminal():
            next_q = 0.0
        else:
            next_state_idx = state_to_index(next_state, self.env)
            next_action_idx = ACTION_TO_INDEX[next_action]
            next_q = self.Q[next_state_idx, next_action_idx].item()

        # SARSA update
        current_q = self.Q[state_idx, action_idx]
        target = reward + self.gamma * next_q
        self.Q[state_idx, action_idx] = current_q + self.alpha * (target - current_q)


class ExpectedSARSAAgent(BaseAgent):
    """
    Agent using Expected SARSA (more stable on-policy learning) with PyTorch GPU acceleration.
    """

    def __init__(self, env, gamma=0.9, alpha=0.1, epsilon=0.5):
        """
        Constructor of the class.

        Args:
            env: Environment instance
            gamma: Discount factor
            alpha: Learning rate
            epsilon: Exploration rate
        """
        super().__init__(env, gamma, alpha, epsilon)
        
        # Q-function as PyTorch tensor
        n_states = get_state_space_size(env)
        n_actions = len(ACTIONS)
        self.Q = torch.zeros(n_states, n_actions, device=self.device, dtype=torch.float32)
        self.default_q = 0.0

    def greedy_action(self, state, allowed_actions):
        """
        Get the greedy action with highest Q-value.

        Args:
            state: Current state
            allowed_actions: Allowed actions for the given state

        Returns:
            Action with highest Q-value
        """
        state_idx = state_to_index(state, self.env)
        
        q_values = []
        for action in allowed_actions:
            q_val = self.Q[state_idx, ACTION_TO_INDEX[action]].item()
            q_values.append(q_val)
        
        max_q = max(q_values)
        best_actions = [
            action for action, q in zip(allowed_actions, q_values) if q == max_q
        ]
        return np.random.choice(best_actions)

    def update(self, state, action, reward, next_state):
        """
        Update Q-function using Expected SARSA update rule.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
        """
        self.steps += 1
        self.total_reward += reward

        state_idx = state_to_index(state, self.env)
        action_idx = ACTION_TO_INDEX[action]
        
        # Expected SARSA: use expected value over all next actions
        if self.env.is_terminal():
            expected_next_q = 0.0
        else:
            next_state_idx = state_to_index(next_state, self.env)
            next_allowed = self.env.get_allowed_actions(next_state)
            
            # Get Q-values for all allowed actions using PyTorch
            next_action_indices = [ACTION_TO_INDEX[a] for a in next_allowed]
            q_values_tensor = self.Q[next_state_idx, next_action_indices]
            
            # Calculate expected value under epsilon-greedy policy
            max_q = q_values_tensor.max().item()
            n_actions = len(next_allowed)
            
            expected_next_q = 0.0
            for i, action in enumerate(next_allowed):
                q_val = q_values_tensor[i].item()
                if q_val == max_q:
                    # Greedy action gets (1 - epsilon + epsilon/n) probability
                    prob = (1 - self.epsilon) + self.epsilon / n_actions
                else:
                    # Non-greedy actions get epsilon/n probability
                    prob = self.epsilon / n_actions
                expected_next_q += prob * q_val

        # Expected SARSA update
        current_q = self.Q[state_idx, action_idx]
        target = reward + self.gamma * expected_next_q
        self.Q[state_idx, action_idx] = current_q + self.alpha * (target - current_q)


# Legacy alias for backwards compatibility
Agent = MonteCarloAgent
