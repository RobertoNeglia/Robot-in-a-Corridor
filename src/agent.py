"""
Agent classes implementing various reinforcement learning algorithms.
"""

import numpy as np
from .utils import ORIENTATION, ACTIONS, TAKE_A_STEP, MOVE


class BaseAgent:
    """
    Base class for reinforcement learning agents.
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
    Agent using Monte Carlo learning with state-value function.
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
        # value function
        self.V = {}
        self.init_V()
        # history
        self.trajectory = []

    def init_V(self):
        """
        Initialize the value function.
        """
        for y in range(1, self.env.corridor.shape[0] - 1):
            for x in range(1, self.env.corridor.shape[1]):
                for orientation in ORIENTATION:
                    self.V[(y, x, orientation)] = np.random.uniform(0.01, 1)

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
        best_value = -np.inf
        # compute the value of each action and return the best one
        for action in allowed_actions:
            y, x, orientation = state
            if action == ACTIONS.TURN_LEFT:
                orientation = ORIENTATION(((orientation.value + 1) % 4))
            elif action == ACTIONS.TURN_RIGHT:
                orientation = ORIENTATION(((orientation.value - 1) % 4))
            else:
                y += TAKE_A_STEP[orientation][0] * MOVE[action.name]
                x += TAKE_A_STEP[orientation][1] * MOVE[action.name]
            value = self.V[(y, x, orientation)]
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
        Update the value function using the Monte Carlo method.
        """
        g = 0
        for state, reward in reversed(self.trajectory):
            g = self.gamma * g + reward
            self.V[state] += self.alpha * (g - self.V[state])
        # reset the history
        self.trajectory = []
        self.reset_episode()
        self.decay_epsilon()


class QLearningAgent(BaseAgent):
    """
    Agent using Q-Learning (off-policy TD control).
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
        # Q-function (action-value function)
        self.Q = {}
        self.default_q = 0.0  # Initialize to 0
        self.init_Q()

    def init_Q(self):
        """
        Initialize the Q-function for all state-action pairs to 0.
        """
        for y in range(1, self.env.corridor.shape[0] - 1):
            for x in range(1, self.env.corridor.shape[1]):
                for orientation in ORIENTATION:
                    state = (y, x, orientation)
                    allowed_actions = self.env.get_allowed_actions(state)
                    for action in allowed_actions:
                        self.Q[(state, action)] = self.default_q

    def greedy_action(self, state, allowed_actions):
        """
        Get the greedy action with highest Q-value.

        Args:
            state: Current state
            allowed_actions: Allowed actions for the given state

        Returns:
            Action with highest Q-value
        """
        q_values = [
            self.Q.get((state, action), self.default_q) for action in allowed_actions
        ]
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

        # Q-Learning: off-policy update using max Q-value of next state
        if self.env.is_terminal():
            # Terminal state has no future value
            max_next_q = 0
        else:
            next_allowed = self.env.get_allowed_actions(next_state)
            max_next_q = max(
                [self.Q.get((next_state, a), self.default_q) for a in next_allowed]
            )

        # Q-Learning update
        current_q = self.Q.get((state, action), self.default_q)
        self.Q[(state, action)] = current_q + self.alpha * (
            reward + self.gamma * max_next_q - current_q
        )


class SARSAAgent(BaseAgent):
    """
    Agent using SARSA (on-policy TD control).
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
        # Q-function (action-value function)
        self.Q = {}
        self.default_q = 0.0  # Initialize to 0
        self.init_Q()

    def init_Q(self):
        """
        Initialize the Q-function for all state-action pairs to 0.
        """
        for y in range(1, self.env.corridor.shape[0] - 1):
            for x in range(1, self.env.corridor.shape[1]):
                for orientation in ORIENTATION:
                    state = (y, x, orientation)
                    allowed_actions = self.env.get_allowed_actions(state)
                    for action in allowed_actions:
                        self.Q[(state, action)] = self.default_q

    def greedy_action(self, state, allowed_actions):
        """
        Get the greedy action with highest Q-value.

        Args:
            state: Current state
            allowed_actions: Allowed actions for the given state

        Returns:
            Action with highest Q-value
        """
        q_values = [
            self.Q.get((state, action), self.default_q) for action in allowed_actions
        ]
        max_q = max(q_values)
        # Handle ties by random selection
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

        # SARSA: on-policy update using Q-value of next state-action pair
        if self.env.is_terminal():
            next_q = 0
        else:
            next_q = self.Q.get((next_state, next_action), self.default_q)

        # SARSA update
        current_q = self.Q.get((state, action), self.default_q)
        self.Q[(state, action)] = current_q + self.alpha * (
            reward + self.gamma * next_q - current_q
        )


class ExpectedSARSAAgent(BaseAgent):
    """
    Agent using Expected SARSA (more stable on-policy learning).
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
        # Q-function (action-value function)
        self.Q = {}
        self.default_q = 0.0  # Initialize to 0
        self.init_Q()

    def init_Q(self):
        """
        Initialize the Q-function for all state-action pairs to 0.
        """
        for y in range(1, self.env.corridor.shape[0] - 1):
            for x in range(1, self.env.corridor.shape[1]):
                for orientation in ORIENTATION:
                    state = (y, x, orientation)
                    allowed_actions = self.env.get_allowed_actions(state)
                    for action in allowed_actions:
                        self.Q[(state, action)] = self.default_q

    def greedy_action(self, state, allowed_actions):
        """
        Get the greedy action with highest Q-value.

        Args:
            state: Current state
            allowed_actions: Allowed actions for the given state

        Returns:
            Action with highest Q-value
        """
        q_values = [
            self.Q.get((state, action), self.default_q) for action in allowed_actions
        ]
        max_q = max(q_values)
        # Handle ties by random selection
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

        # Expected SARSA: use expected value over all next actions
        if self.env.is_terminal():
            expected_next_q = 0
        else:
            next_allowed = self.env.get_allowed_actions(next_state)
            q_values = [
                self.Q.get((next_state, a), self.default_q) for a in next_allowed
            ]

            # Calculate expected value under epsilon-greedy policy
            max_q = max(q_values)
            n_actions = len(next_allowed)

            expected_next_q = 0
            for a, q in zip(next_allowed, q_values):
                if q == max_q:
                    # Greedy action gets (1 - epsilon + epsilon/n) probability
                    prob = (1 - self.epsilon) + self.epsilon / n_actions
                else:
                    # Non-greedy actions get epsilon/n probability
                    prob = self.epsilon / n_actions
                expected_next_q += prob * q

        # Expected SARSA update
        current_q = self.Q.get((state, action), self.default_q)
        self.Q[(state, action)] = current_q + self.alpha * (
            reward + self.gamma * expected_next_q - current_q
        )


# Legacy alias for backwards compatibility
Agent = MonteCarloAgent
