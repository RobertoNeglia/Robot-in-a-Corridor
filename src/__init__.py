"""
Robot-in-a-Corridor: A reinforcement learning project for corridor navigation.
"""

from .environment import Corridor
from .agent import (
    Agent,  # Legacy alias for MonteCarloAgent
    MonteCarloAgent,
    QLearningAgent,
    SARSAAgent,
    ExpectedSARSAAgent,
)
from .utils import ORIENTATION, ACTIONS
from .training import train_monte_carlo, train_td_learning, compare_algorithms

__all__ = [
    "Corridor",
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
]
