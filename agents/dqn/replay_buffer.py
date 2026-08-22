"""
Experience Replay Buffer for Deep Q-Learning.
"""
import random
from collections import deque
from typing import Tuple, List, Dict
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity: int = 50000):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: Dict[str, float],
        action: int,
        reward: float,
        next_state: Dict[str, float],
        done: bool
    ) -> None:
        """Add experience tuple to memory buffer."""
        state_arr = np.array([
            state["y_diff"] / 300.0,
            state["x_diff"] / 400.0,
            state["velocity"] / 10.0,
            state["subsequent_y_diff"] / 300.0
        ], dtype=np.float32)
        next_state_arr = np.array([
            next_state["y_diff"] / 300.0,
            next_state["x_diff"] / 400.0,
            next_state["velocity"] / 10.0,
            next_state["subsequent_y_diff"] / 300.0
        ], dtype=np.float32)
        self.buffer.append((state_arr, action, reward, next_state_arr, done))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample a batch of transitions uniformly at random."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.bool_)
        )

    def __len__(self) -> int:
        return len(self.buffer)
