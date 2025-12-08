"""
Utility functions and constants for the robot corridor environment.
"""

from enum import Enum
import numpy as np

# Directions that the robot can face
ORIENTATION = Enum("ORIENTATION", "UP LEFT DOWN RIGHT", start=0)

# Steps length in each direction
TAKE_A_STEP = {
    ORIENTATION.UP: np.array([-1, 0]),
    ORIENTATION.DOWN: np.array([1, 0]),
    ORIENTATION.RIGHT: np.array([0, 1]),
    ORIENTATION.LEFT: np.array([0, -1]),
}

# Actions that the robot can take
ACTIONS = Enum("ACTIONS", "FORWARD BACKWARD TURN_LEFT TURN_RIGHT")

# Forward and backward actions steps
MOVE = {"FORWARD": 1, "BACKWARD": -1}

# Symbol of the robot in the corridor
ROBOT_SYMBOL = "X"

# Symbol of the goal of the robot in the corridor
GOAL_SYMBOL = "G"

# Symbol of the wall in the corridor
WALL_SYMBOL = "#"

# Symbol of the empty space in the corridor
EMPTY_SYMBOL = " "

# Dictionary that maps symbols to values
SYMBOLS_TO_VALUES = {
    EMPTY_SYMBOL: 0,
    WALL_SYMBOL: -1,
    GOAL_SYMBOL: 1,
    ROBOT_SYMBOL: 2,
}


def get_value_from_symbol(symbol):
    """
    Util function that maps a symbol to a value.

    Args:
        symbol: Symbol to map

    Returns:
        Value mapped from the symbol
    """
    return SYMBOLS_TO_VALUES[symbol]
