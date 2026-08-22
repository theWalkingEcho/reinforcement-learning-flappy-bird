"""
Standalone Evaluation Module for Flappy Bird RL.
Dedicated runner for evaluating trained agents without altering training files or saving checkpoints.
"""
import os
import random
import pygame
from core.interfaces.agent_interface import AgentInterface
from environment.flappy_env import FlappyBirdEnv
from training.metrics_logger import MetricsLogger
from evaluation.eval_renderer import EvalGameRenderer
from utils.config import WindowConfig, QLearningConfig, DQNConfig

class RLEvaluator:
    """
    Dedicated Evaluator for RL agents.
    Executes greedy policy evaluation (eval_mode=True) with read-only checkpoint safety
    and full-window game GUI (Score & Best Score overlays + trajectory path visualization).
    """
    def __init__(self, agent: AgentInterface, mode_name: str = "Q-Learning", model_path: str = None, seed: int = None):
        self.agent = agent
        self.mode_name = mode_name
        self.seed = seed
        self.env = FlappyBirdEnv()
        self.logger = MetricsLogger(agent_name=f"{mode_name} (Eval)")

        if seed is not None:
            random.seed(seed)

        # Determine checkpoint path
        if model_path is None:
            if "tabular" in mode_name.lower() or mode_name.lower() == "q-learning" or mode_name.lower() == "qlearning":
                model_path = QLearningConfig.MODEL_PATH
            else:
                model_path = DQNConfig.MODEL_PATH

        self.model_path = model_path
        if os.path.exists(model_path):
            self.agent.load(model_path)
            print(f"[EVALUATOR] Loaded trained checkpoint from: {model_path}")
        else:
            print(f"[EVALUATOR] Warning: No checkpoint found at {model_path}. Running with initialized weights.")

        self.current_episode = 1
        self.speed_boost = False
        self.paused = False

    def run_gui_evaluation(self, num_episodes: int = 100) -> None:
        """Run interactive Pygame GUI evaluation session with path trajectory visualization."""
        pygame.init()
        window = pygame.display.set_mode((WindowConfig.GAME_WIDTH, WindowConfig.HEIGHT))
        pygame.display.set_caption(f"Flappy Bird RL — {self.mode_name} [EVALUATION MODE]")
        clock = pygame.time.Clock()

        renderer = EvalGameRenderer(window)

        running = True
        while running and self.current_episode <= num_episodes:
            if self.seed is not None:
                random.seed(self.seed + self.current_episode)

            renderer.reset_path()
            state_vec, snapshot = self.env.reset()
            episode_reward = 0.0
            episode_steps = 0
            done = False

            while not done and running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            self.speed_boost = not self.speed_boost
                        elif event.key == pygame.K_p:
                            self.paused = not self.paused

                if self.paused:
                    clock.tick(10)
                    continue

                # Select action in pure greedy evaluation mode (eval_mode=True)
                action = self.agent.select_action(state_vec, eval_mode=True)
                next_state_vec, reward, done, snapshot = self.env.step(action)

                episode_reward += reward
                episode_steps += 1
                state_vec = next_state_vec

                # Render Viewport (Clean Full Game View with Trajectory)
                if not self.speed_boost or episode_steps % 5 == 0:
                    renderer.render(snapshot, best_score=self.logger.best_score, mode_name=self.mode_name)
                    pygame.display.flip()

                    if not self.speed_boost:
                        clock.tick(WindowConfig.FPS)

            self.logger.log_episode(
                episode=self.current_episode,
                score=snapshot.score,
                total_reward=episode_reward,
                steps=episode_steps,
                losses=[],
                epsilon=0.0
            )

            extra_info = ""
            if hasattr(self.agent, "q_table"):
                extra_info = f" | Q-Table Size: {len(self.agent.q_table)}"

            print(
                f"[Eval Ep {self.current_episode:04d}/{num_episodes:04d}] "
                f"Score: {snapshot.score:2d} | "
                f"Reward: {episode_reward:8.1f} | "
                f"Steps: {episode_steps:4d} | "
                f"Policy: Pure Greedy | "
                f"Best: {self.logger.best_score:2d}"
                f"{extra_info}"
            )

            self.current_episode += 1

        pygame.quit()

    def run_headless_evaluation(self, num_episodes: int = 100) -> MetricsLogger:
        """Run fast headless evaluation session without GUI rendering."""
        for ep in range(1, num_episodes + 1):
            if self.seed is not None:
                random.seed(self.seed + ep)

            state_vec, snapshot = self.env.reset()
            episode_reward = 0.0
            episode_steps = 0
            done = False

            while not done:
                action = self.agent.select_action(state_vec, eval_mode=True)
                next_state_vec, reward, done, snapshot = self.env.step(action)
                episode_reward += reward
                episode_steps += 1
                state_vec = next_state_vec

            self.logger.log_episode(
                episode=ep,
                score=snapshot.score,
                total_reward=episode_reward,
                steps=episode_steps,
                losses=[],
                epsilon=0.0
            )

            extra_info = ""
            if hasattr(self.agent, "q_table"):
                extra_info = f" | Q-Table Size: {len(self.agent.q_table)}"

            print(
                f"[Eval Ep {ep:04d}/{num_episodes:04d}] "
                f"Score: {snapshot.score:2d} | "
                f"Reward: {episode_reward:8.1f} | "
                f"Steps: {episode_steps:4d} | "
                f"Policy: Pure Greedy | "
                f"Best: {self.logger.best_score:2d}"
                f"{extra_info}"
            )

        return self.logger
