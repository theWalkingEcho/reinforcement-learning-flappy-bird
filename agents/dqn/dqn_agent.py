"""
Deep Q-Network Agent implementing AgentInterface.
"""
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any
from core.interfaces.agent_interface import AgentInterface
from agents.dqn.network import QNetwork
from agents.dqn.replay_buffer import ReplayBuffer
from utils.config import DQNConfig

class DQNAgent(AgentInterface):
    def __init__(
        self,
        lr: float = DQNConfig.LEARNING_RATE,
        gamma: float = DQNConfig.GAMMA,
        epsilon: float = DQNConfig.EPSILON_START,
        epsilon_min: float = DQNConfig.EPSILON_MIN,
        epsilon_decay: float = DQNConfig.EPSILON_DECAY,
        buffer_capacity: int = DQNConfig.BUFFER_CAPACITY,
        batch_size: int = DQNConfig.BATCH_SIZE,
        target_update_freq: int = DQNConfig.TARGET_UPDATE_FREQ
    ):
        self.gamma = gamma
        self._epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.steps = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Policy & Target Networks
        self.policy_net = QNetwork(state_dim=4, action_dim=2, hidden_dim=DQNConfig.HIDDEN_DIM).to(self.device)
        self.target_net = QNetwork(state_dim=4, action_dim=2, hidden_dim=DQNConfig.HIDDEN_DIM).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()  # Huber Loss
        self.memory = ReplayBuffer(capacity=buffer_capacity)

    @property
    def epsilon(self) -> float:
        return self._epsilon

    def _state_to_tensor(self, state_dict: Dict[str, float]) -> torch.Tensor:
        arr = np.array(
            [state_dict["y_diff"], state_dict["x_diff"], state_dict["velocity"], state_dict["subsequent_y_diff"]],
            dtype=np.float32
        )
        return torch.from_numpy(arr).unsqueeze(0).to(self.device)

    def select_action(self, state: Dict[str, float], eval_mode: bool = False) -> int:
        """Select action via epsilon-greedy policy."""
        if not eval_mode and random.random() < self._epsilon:
            return 1 if random.random() < 0.15 else 0

        state_tensor = self._state_to_tensor(state)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            return int(torch.argmax(q_values, dim=1).item())

    def learn(
        self,
        state: Dict[str, float],
        action: int,
        reward: float,
        next_state: Dict[str, float],
        done: bool
    ) -> float:
        """Push transition to replay buffer and train network if batch size is met."""
        self.memory.push(state, action, reward, next_state, done)
        self.steps += 1

        if len(self.memory) < self.batch_size:
            return 0.0

        # Sample mini-batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Compute Q(s, a)
        current_q = self.policy_net(states_t).gather(1, actions_t)

        # Compute target Q(s', a') using target_net
        with torch.no_grad():
            max_next_q = self.target_net(next_states_t).max(1, keepdim=True)[0]
            target_q = rewards_t + (1.0 - dones_t) * self.gamma * max_next_q

        loss = self.criterion(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Periodically update target network weights
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())

    def update_epsilon(self) -> float:
        """Decay exploration rate."""
        self._epsilon = max(self.epsilon_min, self._epsilon * self.epsilon_decay)
        return self._epsilon

    def save(self, filepath: str = DQNConfig.MODEL_PATH) -> None:
        """Save network state and parameters to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self._epsilon
        }, filepath)

    def load(self, filepath: str = DQNConfig.MODEL_PATH) -> bool:
        """Load network state from disk if file exists."""
        if os.path.exists(filepath):
            try:
                checkpoint = torch.load(filepath, map_location=self.device)
                self.policy_net.load_state_dict(checkpoint["policy_net"])
                self.target_net.load_state_dict(checkpoint["target_net"])
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                self._epsilon = checkpoint.get("epsilon", self._epsilon)
                return True
            except Exception as e:
                print(f"Failed to load DQN model from {filepath}: {e}")
                return False
        return False
