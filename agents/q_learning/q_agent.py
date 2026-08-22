"""
Tabular Q-Learning Agent implementing AgentInterface.
"""
import random
import os
import pickle
from typing import Dict, Tuple, Any
from core.interfaces.agent_interface import AgentInterface
from agents.q_learning.discretizer import StateDiscretizer
from utils.config import QLearningConfig

class QLearningAgent(AgentInterface):
    def __init__(
        self,
        alpha: float = QLearningConfig.ALPHA,
        gamma: float = QLearningConfig.GAMMA,
        epsilon: float = QLearningConfig.EPSILON_START,
        epsilon_min: float = QLearningConfig.EPSILON_MIN,
        epsilon_decay: float = QLearningConfig.EPSILON_DECAY
    ):
        self.alpha = alpha
        self.gamma = gamma
        self._epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.discretizer = StateDiscretizer()
        
        # Q-Table: dict mapping (discrete_state_tuple, action_int) -> float
        self.q_table: Dict[Tuple[Tuple[int, int, int], int], float] = {}

    @property
    def epsilon(self) -> float:
        return self._epsilon

    def get_q_value(self, state_tuple: Tuple[int, int, int], action: int) -> float:
        """Retrieve Q(s, a), default 0.0."""
        return self.q_table.get((state_tuple, action), 0.0)

    def select_action(self, state: Dict[str, float], eval_mode: bool = False) -> int:
        """Select action via epsilon-greedy policy."""
        discrete_state = self.discretizer.discretize(state)
        
        if not eval_mode and random.random() < self._epsilon:
            return 1 if random.random() < 0.15 else 0

        q0 = self.get_q_value(discrete_state, 0)
        q1 = self.get_q_value(discrete_state, 1)

        if q1 > q0:
            return 1
        elif q0 > q1:
            return 0
        else:
            # Smart fallback for unvisited states (Q(s,0) == Q(s,1)):
            # If target gap is above bird (y_diff < 0), flap (1); otherwise fall (0)
            return 1 if state.get("y_diff", 0) < -15.0 else 0

    def learn(
        self,
        state: Dict[str, float],
        action: int,
        reward: float,
        next_state: Dict[str, float],
        done: bool
    ) -> float:
        """Update Q-table entry using TD learning target."""
        s = self.discretizer.discretize(state)
        s_next = self.discretizer.discretize(next_state)

        current_q = self.get_q_value(s, action)

        if done:
            max_next_q = 0.0
        else:
            max_next_q = max(self.get_q_value(s_next, 0), self.get_q_value(s_next, 1))

        td_target = reward + self.gamma * max_next_q
        td_error = td_target - current_q
        new_q = current_q + self.alpha * td_error

        self.q_table[(s, action)] = new_q
        return abs(td_error)

    def update_epsilon(self) -> float:
        """Decay exploration rate."""
        self._epsilon = max(self.epsilon_min, self._epsilon * self.epsilon_decay)
        return self._epsilon

    def save(self, filepath: str = QLearningConfig.MODEL_PATH) -> None:
        """Save Q-table to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            "q_table": self.q_table,
            "epsilon": self._epsilon
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    def load(self, filepath: str = QLearningConfig.MODEL_PATH) -> bool:
        """Load Q-table from disk if file exists."""
        if os.path.exists(filepath):
            try:
                with open(filepath, "rb") as f:
                    data = pickle.load(f)
                self.q_table = data.get("q_table", {})
                self._epsilon = data.get("epsilon", self._epsilon)
                return True
            except Exception as e:
                print(f"Failed to load Q-Table from {filepath}: {e}")
                return False
        return False
