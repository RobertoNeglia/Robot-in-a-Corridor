"""
Corridor environment for robot navigation.
"""

import numpy as np
import matplotlib.pyplot as plt
from .utils import (
    ORIENTATION,
    ACTIONS,
    TAKE_A_STEP,
    MOVE,
    ROBOT_SYMBOL,
    GOAL_SYMBOL,
    WALL_SYMBOL,
    EMPTY_SYMBOL,
    SYMBOLS_TO_VALUES,
)


class Corridor:
    """
    Class that represents the corridor environment.
    """

    def __init__(self, length=10, width=5):
        """
        Constructor of the class.

        Args:
            length: Length of the corridor
            width: Width of the corridor
        """
        self.length = length + 3
        self.width = width + 2
        # initialize the corridor
        self.corridor = np.empty((self.width, self.length), dtype=str)
        self.corridor[:] = EMPTY_SYMBOL
        # initial state of the agent: a state is defined by (y, x, orientation)
        self.agent_pos = np.array([3, 1])
        self.agent_orientation = ORIENTATION.RIGHT
        self.corridor[self.agent_pos[0], self.agent_pos[1]] = ROBOT_SYMBOL
        # define walls
        self.corridor[:, 0] = WALL_SYMBOL
        self.corridor[0, :] = WALL_SYMBOL
        self.corridor[self.width - 1, :] = WALL_SYMBOL
        # define the goal
        self.corridor[1 : self.width - 1, self.length - 1] = GOAL_SYMBOL
        # data structure that contains the allowed actions for each state
        self.allowed_actions = {}
        self.init_allowed_actions()

    def is_allowed_action(self, state, action):
        """
        Check if an action is allowed in a given state.

        Args:
            state: State to check
            action: Action to check

        Returns:
            True if the action is allowed, False otherwise
        """
        if action == ACTIONS.TURN_LEFT or action == ACTIONS.TURN_RIGHT:
            return True

        y, x, orientation = state
        y += TAKE_A_STEP[orientation][0] * MOVE[action.name]
        x += TAKE_A_STEP[orientation][1] * MOVE[action.name]

        if y < 0 or y > (self.width - 1) or x < 0 or x > (self.length - 1):
            return False

        if self.corridor[y, x] == WALL_SYMBOL:
            return False
        else:
            return True

    def init_allowed_actions(self):
        """
        Initialize the data structure that contains the allowed actions for each state.
        """
        allowed_actions = {}
        for y, row in enumerate(self.corridor):
            for x, cell in enumerate(row):
                if cell != WALL_SYMBOL:
                    for orientation in ORIENTATION:
                        allowed_actions[(y, x, orientation)] = []
                        for action in ACTIONS:
                            if self.is_allowed_action(((y, x, orientation)), action):
                                allowed_actions[(y, x, orientation)].append(action)

        self.allowed_actions = allowed_actions

    def update(self, action):
        """
        Update the environment given an action.

        Args:
            action: Action to take
        """
        if action in self.allowed_actions[self.get_state()]:
            if action == ACTIONS.TURN_LEFT:
                self.agent_orientation = ORIENTATION(
                    ((self.agent_orientation.value + 1) % 4)
                )
            elif action == ACTIONS.TURN_RIGHT:
                self.agent_orientation = ORIENTATION(
                    ((self.agent_orientation.value - 1) % 4)
                )
            else:
                if self.agent_pos[1] == self.length - 1:
                    replace = GOAL_SYMBOL
                else:
                    replace = EMPTY_SYMBOL
                self.corridor[self.agent_pos[0], self.agent_pos[1]] = replace
                self.agent_pos += (
                    TAKE_A_STEP[self.agent_orientation] * MOVE[action.name]
                )
                self.corridor[self.agent_pos[0], self.agent_pos[1]] = ROBOT_SYMBOL

    def give_reward(self):
        """
        Give the reward to the agent.

        Returns:
            Reward value
        """
        if self.agent_pos[1] == self.length - 1:
            return 100
        else:
            return -1

    def is_terminal(self):
        """
        Check if the agent is in a terminal state.

        Returns:
            True if the agent is in a terminal state, False otherwise
        """
        return self.agent_pos[1] == self.length - 1

    def plot_corridor(self):
        """
        Plot the corridor using matplotlib.
        """
        plt.figure(figsize=(30, 30))
        res = np.vectorize(SYMBOLS_TO_VALUES.get)(self.corridor)
        plt.imshow(res, cmap=plt.cm.gray, interpolation="nearest")
        plt.xticks([])
        plt.yticks([])

    def get_state(self):
        """
        Get the current state of the agent.
        The current state is defined as a tuple containing (y, x, orientation).

        Returns:
            Current state of the agent
        """
        return (self.agent_pos[0], self.agent_pos[1], self.agent_orientation)

    def get_state_and_reward(self):
        """
        Get the current state of the agent and the reward.

        Returns:
            A tuple containing the current state of the agent and the reward
        """
        return (self.get_state(), self.give_reward())

    def get_allowed_actions(self, state):
        """
        Get the allowed actions for a given state.

        Args:
            state: State to check

        Returns:
            Allowed actions for the given state
        """
        return self.allowed_actions[state]

    def reset_env(self):
        """
        Resets the environment.
        """
        self.corridor[:] = EMPTY_SYMBOL
        # initial state of the agent: a state is defined by (y, x, orientation)
        self.agent_pos = np.array([3, 1])
        self.agent_orientation = ORIENTATION.RIGHT
        self.corridor[self.agent_pos[0], self.agent_pos[1]] = ROBOT_SYMBOL
        # define walls
        self.corridor[:, 0] = WALL_SYMBOL
        self.corridor[0, :] = WALL_SYMBOL
        self.corridor[self.width - 1, :] = WALL_SYMBOL
        # define the goal
        self.corridor[1 : self.width - 1, self.length - 1] = GOAL_SYMBOL
