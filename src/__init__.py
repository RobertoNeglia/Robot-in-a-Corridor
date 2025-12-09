"""
Robot-in-a-Corridor: A reinforcement learning project for corridor navigation.
"""

from .environment import Corridor, Maze, Rooms, GridWorld, BaseEnvironment
from .agent import (
    Agent,  # Legacy alias for MonteCarloAgent
    MonteCarloAgent,
    QLearningAgent,
    SARSAAgent,
    ExpectedSARSAAgent,
)
from .utils import ORIENTATION, ACTIONS
from .training import train_monte_carlo, train_td_learning, compare_algorithms
from .visualization import (
    visualize_training_progress,
    visualize_training_progress_overlay,
    visualize_training_progress_animated_overlay,
)

__all__ = [
    "Corridor",
    "Maze",
    "Rooms",
    "GridWorld",
    "BaseEnvironment",
    "Agent",
    "MonteCarloAgent",
    "QLearningAgent",
    "SARSAAgent",
    "ExpectedSARSAAgent",
    "ORIENTATION",
    "ACTIONS",
    "train_monte_carlo",
    "train_td_learning",
    "compare_algorithms",
    "visualize_training_progress",
    "visualize_training_progress_overlay",
    "visualize_training_progress_animated_overlay",
]
