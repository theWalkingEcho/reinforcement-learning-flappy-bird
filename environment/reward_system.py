"""
Reward & Penalty System calculation for Flappy Bird RL environment.
Encapsulates single responsibility of reward shaping.
"""
from core.entities.bird import Bird
from core.entities.pipe import PipePair
from utils.config import RewardConfig, PhysicsConfig

class RewardSystem:
    @staticmethod
    def calculate_reward(
        bird: Bird,
        next_pipe: PipePair | None,
        pipe_cleared: bool,
        is_crash: bool
    ) -> float:
        """Calculate shaped reward for current step transition."""
        if is_crash:
            return RewardConfig.CRASH_PENALTY

        reward = RewardConfig.SURVIVAL

        if pipe_cleared:
            reward += RewardConfig.PIPE_CLEARED

        # Centering Shaping: reward bird for staying aligned with upcoming pipe gap center
        if next_pipe is not None:
            dist_to_center = abs(bird.y - next_pipe.gap_center_y)
            max_dist = next_pipe.gap_height / 2.0 + 50.0
            shaping = max(0.0, 1.0 - (dist_to_center / max_dist))
            reward += RewardConfig.CENTERING_SHAPING * shaping

        return reward
