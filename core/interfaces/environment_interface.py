"""
Abstract Environment Interface defining contract for the Flappy Bird game environment.
"""
from abc import ABC, abstractmethod
from typing import Tuple, Any
from core.entities.game_state import GameState

class EnvironmentInterface(ABC):
    @abstractmethod
    def reset(self) -> Tuple[Any, GameState]:
        """Reset game state (bird, pipes, score) and return initial state vector and snapshot."""
        pass

    @abstractmethod
    def step(self, action: int) -> Tuple[Any, float, bool, GameState]:
        """Advance game state by one frame with action (0 or 1). Returns (next_state, reward, done, game_snapshot)."""
        pass

    @abstractmethod
    def get_state_vector(self) -> Any:
        """Return numerical feature vector representation of current state."""
        pass
