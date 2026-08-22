"""
Abstract Agent Interface defining contract for RL Agents (Interface Segregation / Dependency Inversion).
"""
from abc import ABC, abstractmethod
from typing import Any, Tuple

class AgentInterface(ABC):
    @abstractmethod
    def select_action(self, state: Any, eval_mode: bool = False) -> int:
        """Select an action (0: Do Nothing, 1: Jump) given the current state."""
        pass

    @abstractmethod
    def learn(self, state: Any, action: int, reward: float, next_state: Any, done: bool) -> float:
        """Process transition step and update internal model/policy. Returns step loss or 0.0."""
        pass

    @abstractmethod
    def update_epsilon(self) -> float:
        """Decay exploration rate epsilon and return new value."""
        pass

    @abstractmethod
    def save(self, filepath: str) -> None:
        """Persist agent weights / Q-table to disk."""
        pass

    @abstractmethod
    def load(self, filepath: str) -> bool:
        """Load agent weights / Q-table from disk if exists."""
        pass

    @property
    @abstractmethod
    def epsilon(self) -> float:
        """Get current epsilon value."""
        pass
