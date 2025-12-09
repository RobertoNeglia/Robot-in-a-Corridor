"""
Environment classes for robot navigation.
Includes multiple environment types: Corridor, Maze, Rooms, and GridWorld.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
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


def has_path(grid, start, goal):
    """
    Check if there's a valid path from start to goal using BFS.

    Args:
        grid: 2D numpy array representing the environment
        start: Tuple (y, x) of start position
        goal: Tuple (y, x) of goal position

    Returns:
        True if path exists, False otherwise
    """
    if grid[start[0], start[1]] == WALL_SYMBOL or grid[goal[0], goal[1]] == WALL_SYMBOL:
        return False

    visited = set()
    queue = deque([start])
    visited.add(start)

    while queue:
        y, x = queue.popleft()

        if (y, x) == goal:
            return True

        # Check all 4 directions
        for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            ny, nx = y + dy, x + dx

            if (
                (ny, nx) not in visited
                and 0 <= ny < grid.shape[0]
                and 0 <= nx < grid.shape[1]
            ):
                if grid[ny, nx] != WALL_SYMBOL:
                    visited.add((ny, nx))
                    queue.append((ny, nx))

    return False


class BaseEnvironment:
    """
    Base class for navigation environments.
    """

    def __init__(self):
        """Base constructor - to be called by subclasses."""
        self.corridor = None
        self.agent_pos = None
        self.agent_orientation = None
        self.allowed_actions = {}
        self.length = None
        self.width = None

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
                if self.corridor[self.agent_pos[0], self.agent_pos[1]] == GOAL_SYMBOL:
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
        Give the reward to the agent (to be overridden by subclasses if needed).

        Returns:
            Reward value
        """
        if self.is_terminal():
            return 100
        else:
            return -1

    def is_terminal(self):
        """
        Check if the agent is in a terminal state (to be overridden by subclasses).

        Returns:
            True if the agent is in a terminal state, False otherwise
        """
        raise NotImplementedError("Must be implemented by subclass")

    def plot_corridor(self):
        """
        Plot the environment using matplotlib.
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
        Resets the environment (to be overridden by subclasses).
        """
        raise NotImplementedError("Must be implemented by subclass")


class Corridor(BaseEnvironment):
    """
    Simple corridor environment - agent navigates from left to right.
    """

    def __init__(self, length=10, width=5):
        """
        Constructor of the class.

        Args:
            length: Length of the corridor
            width: Width of the corridor
        """
        super().__init__()
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
        self.init_allowed_actions()

    def is_terminal(self):
        """
        Check if the agent is in a terminal state.

        Returns:
            True if the agent is in a terminal state, False otherwise
        """
        return self.agent_pos[1] == self.length - 1

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


class Maze(BaseEnvironment):
    """
    Maze environment with obstacles - agent must navigate around walls to reach goal.
    """

    def __init__(self, size=15, obstacle_density=0.2, seed=None):
        """
        Constructor of the maze environment.

        Args:
            size: Size of the square maze (size x size)
            obstacle_density: Fraction of cells to fill with obstacles (0.0 to 1.0)
            seed: Random seed for reproducible mazes
        """
        super().__init__()
        if seed is not None:
            np.random.seed(seed)

        self.length = size
        self.width = size
        self.goal_pos = (size - 2, size - 2)

        # Generate maze with guaranteed path
        max_attempts = 100
        for attempt in range(max_attempts):
            self.corridor = np.empty((self.width, self.length), dtype=str)
            self.corridor[:] = EMPTY_SYMBOL

            # Add border walls
            self.corridor[0, :] = WALL_SYMBOL
            self.corridor[-1, :] = WALL_SYMBOL
            self.corridor[:, 0] = WALL_SYMBOL
            self.corridor[:, -1] = WALL_SYMBOL

            # Add random obstacles
            inner_cells = (size - 2) * (size - 2)
            n_obstacles = int(inner_cells * obstacle_density)

            for _ in range(n_obstacles):
                y = np.random.randint(1, size - 1)
                x = np.random.randint(1, size - 1)
                # Don't place obstacle at start or goal positions
                if (y, x) != (1, 1) and (y, x) != self.goal_pos:
                    self.corridor[y, x] = WALL_SYMBOL

            # Check if there's a valid path
            if has_path(self.corridor, (1, 1), self.goal_pos):
                break

            if attempt == max_attempts - 1:
                # If we couldn't generate a valid maze, create a simple one
                self.corridor[:] = EMPTY_SYMBOL
                self.corridor[0, :] = WALL_SYMBOL
                self.corridor[-1, :] = WALL_SYMBOL
                self.corridor[:, 0] = WALL_SYMBOL
                self.corridor[:, -1] = WALL_SYMBOL

        # Set start position (top-left)
        self.agent_pos = np.array([1, 1])
        self.agent_orientation = ORIENTATION.RIGHT
        self.corridor[self.agent_pos[0], self.agent_pos[1]] = ROBOT_SYMBOL

        # Set goal (bottom-right)
        self.corridor[self.goal_pos[0], self.goal_pos[1]] = GOAL_SYMBOL

        self.init_allowed_actions()

    def is_terminal(self):
        """Check if agent reached the goal."""
        return tuple(self.agent_pos) == self.goal_pos

    def reset_env(self):
        """Reset to initial state."""
        self.corridor[self.corridor == ROBOT_SYMBOL] = EMPTY_SYMBOL
        self.agent_pos = np.array([1, 1])
        self.agent_orientation = ORIENTATION.RIGHT
        self.corridor[self.agent_pos[0], self.agent_pos[1]] = ROBOT_SYMBOL
        self.corridor[self.goal_pos[0], self.goal_pos[1]] = GOAL_SYMBOL


class Rooms(BaseEnvironment):
    """
    Multi-room environment - agent must navigate through connected rooms to reach goal.
    """

    def __init__(self, n_rooms=4, room_size=8):
        """
        Constructor of the rooms environment.

        Args:
            n_rooms: Number of rooms (1, 4, or 9 for 1x1, 2x2, or 3x3 grid)
            room_size: Size of each room
        """
        super().__init__()

        if n_rooms == 1:
            grid_size = 1
        elif n_rooms == 4:
            grid_size = 2
        elif n_rooms == 9:
            grid_size = 3
        else:
            raise ValueError("n_rooms must be 1, 4, or 9")

        total_size = grid_size * room_size + grid_size + 1
        self.length = total_size
        self.width = total_size
        self.corridor = np.empty((self.width, self.length), dtype=str)
        self.corridor[:] = EMPTY_SYMBOL

        # Add outer walls
        self.corridor[0, :] = WALL_SYMBOL
        self.corridor[-1, :] = WALL_SYMBOL
        self.corridor[:, 0] = WALL_SYMBOL
        self.corridor[:, -1] = WALL_SYMBOL

        # Add room dividing walls with doorways
        # Strategy: Create doorways that ensure connectivity between adjacent rooms
        for i in range(1, grid_size):
            wall_pos = i * (room_size + 1)

            # Horizontal wall (divides rooms vertically)
            self.corridor[wall_pos, :] = WALL_SYMBOL
            # Add doorways in each segment of the horizontal wall
            for j in range(grid_size):
                door_x = j * (room_size + 1) + room_size // 2 + 1
                self.corridor[wall_pos, door_x] = EMPTY_SYMBOL

            # Vertical wall (divides rooms horizontally)
            self.corridor[:, wall_pos] = WALL_SYMBOL
            # Add doorways in each segment of the vertical wall
            for j in range(grid_size):
                door_y = j * (room_size + 1) + room_size // 2 + 1
                self.corridor[door_y, wall_pos] = EMPTY_SYMBOL

        # Set start position (top-left room)
        start_pos = (2, 2)
        self.agent_pos = np.array(start_pos)
        self.agent_orientation = ORIENTATION.RIGHT

        # Set goal (bottom-right room)
        self.goal_pos = (total_size - 3, total_size - 3)

        # Verify path exists - should always exist with proper doorway placement
        if not has_path(self.corridor, start_pos, self.goal_pos):
            # Safety fallback: if somehow path doesn't exist, add more doorways
            for i in range(1, grid_size):
                wall_pos = i * (room_size + 1)
                # Add additional central doorways
                center = total_size // 2
                self.corridor[wall_pos, center] = EMPTY_SYMBOL
                self.corridor[center, wall_pos] = EMPTY_SYMBOL

        self.corridor[self.agent_pos[0], self.agent_pos[1]] = ROBOT_SYMBOL
        self.corridor[self.goal_pos[0], self.goal_pos[1]] = GOAL_SYMBOL

        self.init_allowed_actions()

    def is_terminal(self):
        """Check if agent reached the goal."""
        return tuple(self.agent_pos) == self.goal_pos

    def reset_env(self):
        """Reset to initial state."""
        self.corridor[self.corridor == ROBOT_SYMBOL] = EMPTY_SYMBOL
        self.agent_pos = np.array([2, 2])
        self.agent_orientation = ORIENTATION.RIGHT
        self.corridor[self.agent_pos[0], self.agent_pos[1]] = ROBOT_SYMBOL
        self.corridor[self.goal_pos[0], self.goal_pos[1]] = GOAL_SYMBOL


class GridWorld(BaseEnvironment):
    """
    Open grid world - agent can move freely in any direction to reach goal.
    """

    def __init__(self, size=10):
        """
        Constructor of the grid world environment.

        Args:
            size: Size of the square grid (size x size)
        """
        super().__init__()

        self.length = size
        self.width = size
        self.corridor = np.empty((self.width, self.length), dtype=str)
        self.corridor[:] = EMPTY_SYMBOL

        # Add border walls
        self.corridor[0, :] = WALL_SYMBOL
        self.corridor[-1, :] = WALL_SYMBOL
        self.corridor[:, 0] = WALL_SYMBOL
        self.corridor[:, -1] = WALL_SYMBOL

        # Set start position (center-left)
        start_y = size // 2
        self.agent_pos = np.array([start_y, 1])
        self.agent_orientation = ORIENTATION.RIGHT
        self.corridor[self.agent_pos[0], self.agent_pos[1]] = ROBOT_SYMBOL

        # Set goal (center-right)
        self.goal_pos = (start_y, size - 2)
        self.corridor[self.goal_pos[0], self.goal_pos[1]] = GOAL_SYMBOL

        self.init_allowed_actions()

    def is_terminal(self):
        """Check if agent reached the goal."""
        return tuple(self.agent_pos) == self.goal_pos

    def reset_env(self):
        """Reset to initial state."""
        self.corridor[self.corridor == ROBOT_SYMBOL] = EMPTY_SYMBOL
        start_y = self.width // 2
        self.agent_pos = np.array([start_y, 1])
        self.agent_orientation = ORIENTATION.RIGHT
        self.corridor[self.agent_pos[0], self.agent_pos[1]] = ROBOT_SYMBOL
        self.corridor[self.goal_pos[0], self.goal_pos[1]] = GOAL_SYMBOL
