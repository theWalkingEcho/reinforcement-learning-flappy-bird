"""
Persistence utilities for loading and saving trained Q-tables and PyTorch DQN models.
"""
from core.interfaces.agent_interface import AgentInterface
from utils.config import QLearningConfig, DQNConfig

class ModelPersistenceManager:
    @staticmethod
    def save_agent(agent: AgentInterface, mode: str) -> None:
        """Save agent based on algorithm mode."""
        if mode.lower() in ("qlearning", "q_learning", "q", "q-learning"):
            agent.save(QLearningConfig.MODEL_PATH)
        elif mode.lower() in ("dqn", "deep_q", "deep q-learning (dqn)"):
            agent.save(DQNConfig.MODEL_PATH)

    @staticmethod
    def load_agent(agent: AgentInterface, mode: str) -> bool:
        """Load agent weights/table if saved checkpoint exists."""
        if mode.lower() in ("qlearning", "q_learning", "q", "q-learning"):
            return agent.load(QLearningConfig.MODEL_PATH)
        elif mode.lower() in ("dqn", "deep_q", "deep q-learning (dqn)"):
            return agent.load(DQNConfig.MODEL_PATH)
        return False
