"""
FlappyBird Environment implementing EnvironmentInterface.
Manages game dynamics, collision detection, pipe spawning, state vector extraction.
"""
from typing import Tuple, List, Dict
from core.interfaces.environment_interface import EnvironmentInterface
from core.entities.bird import Bird
from core.entities.pipe import PipePair
from core.entities.game_state import GameState
from environment.reward_system import RewardSystem
from utils.config import PhysicsConfig, WindowConfig

class FlappyBirdEnv(EnvironmentInterface):
    def __init__(self):
        self.bird = Bird()
        self.pipes: List[PipePair] = []
        self.score = 0
        self.frame_count = 0
        self.is_game_over = False
        self.reset()

    def reset(self) -> Tuple[Dict[str, float], GameState]:
        """Reset environment state for a new episode."""
        self.bird.reset()
        self.pipes.clear()
        # Spawn initial pipes
        first_pipe_x = WindowConfig.GAME_WIDTH + 100
        self.pipes.append(PipePair.create_random(first_pipe_x))
        self.pipes.append(PipePair.create_random(first_pipe_x + PhysicsConfig.PIPE_SPACING))
        
        self.score = 0
        self.frame_count = 0
        self.is_game_over = False
        
        snapshot = self.get_game_snapshot()
        state_vec = self.get_state_vector()
        return state_vec, snapshot

    def step(self, action: int) -> Tuple[Dict[str, float], float, bool, GameState]:
        """
        Execute action:
        action 1: Bird flap / jump
        action 0: Do nothing (fall under gravity)
        """
        if self.is_game_over:
            state_vec = self.get_state_vector()
            snapshot = self.get_game_snapshot()
            return state_vec, 0.0, True, snapshot

        self.frame_count += 1

        # 1. Apply action
        if action == 1:
            self.bird.jump()

        # 2. Physics update
        self.bird.update()

        # 3. Update pipes & handle cleared pipes
        pipe_cleared = False
        for pipe in self.pipes:
            pipe.update()
            # Check if bird passed pipe
            if not pipe.passed and (pipe.x + pipe.width) < self.bird.x:
                pipe.passed = True
                self.score += 1
                pipe_cleared = True

        # Remove offscreen pipes and spawn new ones
        if self.pipes and (self.pipes[0].x + self.pipes[0].width) < 0:
            self.pipes.pop(0)
            last_pipe_x = self.pipes[-1].x if self.pipes else WindowConfig.GAME_WIDTH
            self.pipes.append(PipePair.create_random(last_pipe_x + PhysicsConfig.PIPE_SPACING))

        # 4. Collision check
        is_crash = self._check_collision()
        if is_crash:
            self.is_game_over = True

        # 5. Calculate reward
        snapshot = self.get_game_snapshot()
        next_pipe = snapshot.next_pipe
        reward = RewardSystem.calculate_reward(self.bird, next_pipe, pipe_cleared, is_crash)

        state_vec = self.get_state_vector()
        return state_vec, reward, self.is_game_over, snapshot

    def _check_collision(self) -> bool:
        """Check collision with top ceiling, floor, or pipe pairs."""
        # Ceiling or Floor
        if self.bird.y - self.bird.radius <= 0:
            return True
        if self.bird.y + self.bird.radius >= WindowConfig.HEIGHT:
            return True

        # Pipes
        bird_box = self.bird.bounding_box
        for pipe in self.pipes:
            if pipe.collides_with_bird(bird_box):
                return True

        return False

    def get_state_vector(self) -> Dict[str, float]:
        """
        Extract numeric state features:
        - y_diff: Vertical distance to next pipe gap center
        - x_diff: Horizontal distance to next pipe front edge
        - velocity: Bird vertical velocity
        - subsequent_y_diff: Vertical distance to subsequent pipe gap center
        """
        snapshot = self.get_game_snapshot()
        next_p = snapshot.next_pipe
        subsequent_p = snapshot.subsequent_pipe

        if next_p is not None:
            y_diff = next_p.gap_center_y - self.bird.y
            x_diff = next_p.x - self.bird.x
        else:
            y_diff = 0.0
            x_diff = WindowConfig.GAME_WIDTH

        if subsequent_p is not None:
            subsequent_y_diff = subsequent_p.gap_center_y - self.bird.y
        else:
            subsequent_y_diff = y_diff

        return {
            "y_diff": y_diff,
            "x_diff": x_diff,
            "velocity": self.bird.velocity,
            "subsequent_y_diff": subsequent_y_diff
        }

    def get_game_snapshot(self) -> GameState:
        """Return immutable current state snapshot."""
        return GameState(
            bird=Bird(x=self.bird.x, y=self.bird.y, velocity=self.bird.velocity, radius=self.bird.radius),
            pipes=[PipePair(p.x, p.gap_center_y, p.gap_height, p.width, p.passed) for p in self.pipes],
            score=self.score,
            is_game_over=self.is_game_over,
            frame_count=self.frame_count
        )
