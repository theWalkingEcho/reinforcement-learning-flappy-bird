"""
Training Orchestrator for running training loops, UI rendering, event handling.
"""
import os
import sys
import pygame
from core.interfaces.agent_interface import AgentInterface
from environment.flappy_env import FlappyBirdEnv
from training.metrics_logger import MetricsLogger
from ui.renderer import GameRenderer
from ui.dashboard import MetricsDashboard
from utils.persistence import ModelPersistenceManager
from utils.config import WindowConfig, RewardConfig, LOGS_DIR

class RLTrainer:
    def __init__(self, agent: AgentInterface, mode_name: str = "Q-Learning"):
        self.agent = agent
        self.mode_name = mode_name
        self.env = FlappyBirdEnv()
        self.logger = MetricsLogger(agent_name=mode_name)
        
        # Load pre-trained checkpoint if available
        loaded = ModelPersistenceManager.load_agent(self.agent, self.mode_name)
        if loaded:
            print(f"Loaded existing checkpoint for {self.mode_name} (epsilon: {self.agent.epsilon:.3f})")

        self.current_episode = 1
        self.speed_boost = False
        self.paused = False

    def run_gui_training(self, num_episodes: int = 5000) -> None:
        """Run training loop with Pygame visual viewport and live metrics dashboard."""
        pygame.init()
        window = pygame.display.set_mode((WindowConfig.TOTAL_WIDTH, WindowConfig.HEIGHT))
        pygame.display.set_caption(f"Flappy Bird RL — {self.mode_name}")
        clock = pygame.time.Clock()

        # Split Viewport Surfaces
        game_surface = pygame.Surface((WindowConfig.GAME_WIDTH, WindowConfig.HEIGHT))
        dashboard_surface = pygame.Surface((WindowConfig.DASHBOARD_WIDTH, WindowConfig.HEIGHT))

        renderer = GameRenderer(game_surface)
        dashboard = MetricsDashboard(dashboard_surface)

        running = True
        while running and self.current_episode <= num_episodes:
            # 1. Reset Environment for new episode
            state_vec, snapshot = self.env.reset()
            episode_reward = 0.0
            episode_steps = 0
            episode_losses = []
            done = False

            # Episode Loop
            while not done and running:
                # Handle Pygame Events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            self.speed_boost = not self.speed_boost
                        elif event.key == pygame.K_p:
                            self.paused = not self.paused
                        elif event.key == pygame.K_s:
                            ModelPersistenceManager.save_agent(self.agent, self.mode_name)
                            print("Model saved manually!")

                if self.paused:
                    clock.tick(10)
                    continue

                # 2. Select Action & Step Environment
                action = self.agent.select_action(state_vec, eval_mode=False)
                next_state_vec, reward, done, snapshot = self.env.step(action)

                # 3. Agent Learn Step
                loss = self.agent.learn(state_vec, action, reward, next_state_vec, done)
                if loss > 0.0:
                    episode_losses.append(loss)

                episode_reward += reward
                episode_steps += 1
                state_vec = next_state_vec

                # 4. Render Frame (Skip frames if speed boost active for fast training)
                if not self.speed_boost or episode_steps % 5 == 0:
                    renderer.render(snapshot, agent_name=self.mode_name, epsilon=self.agent.epsilon)
                    q_table_size = len(getattr(self.agent, "q_table", {}))
                    dashboard.render(self.logger, self.current_episode, self.mode_name, q_table_size)

                    window.blit(game_surface, (0, 0))
                    window.blit(dashboard_surface, (WindowConfig.DASHBOARD_WIDTH, WindowConfig.HEIGHT))
                    pygame.display.flip()

                    if not self.speed_boost:
                        clock.tick(WindowConfig.FPS)

            # End of Episode logic
            self.agent.update_epsilon()
            metrics = self.logger.log_episode(
                episode=self.current_episode,
                score=snapshot.score,
                total_reward=episode_reward,
                steps=episode_steps,
                losses=episode_losses,
                epsilon=self.agent.epsilon
            )

            # Terminal Progress Log
            extra_info = ""
            if hasattr(self.agent, "q_table"):
                extra_info = f" | Q-Table Size: {len(self.agent.q_table)}"
            elif episode_losses:
                extra_info = f" | Avg Loss: {metrics.avg_loss:.4f}"

            print(
                f"[Episode {self.current_episode:04d}/{num_episodes:04d}] "
                f"Score: {snapshot.score:2d} | "
                f"Reward: {episode_reward:8.1f} | "
                f"Crash Penalty: {RewardConfig.CRASH_PENALTY:6.1f} | "
                f"Steps: {episode_steps:4d} | "
                f"Epsilon: {self.agent.epsilon:.3f} | "
                f"Best: {self.logger.best_score:2d}"
                f"{extra_info}"
            )

            # Auto-save model every 25 episodes
            if self.current_episode % 25 == 0:
                ModelPersistenceManager.save_agent(self.agent, self.mode_name)

            self.current_episode += 1

        # Save final trained weights and metrics
        ModelPersistenceManager.save_agent(self.agent, self.mode_name)
        csv_name = "q_learning_metrics.csv" if ("q-learning" in self.mode_name.lower() or self.mode_name.lower() == "qlearning") else "dqn_metrics.csv"
        self.logger.export_csv(os.path.join(LOGS_DIR, csv_name))
        print(f"Metrics saved to logs/{csv_name}")
        pygame.quit()

    def run_headless_training(self, num_episodes: int = 500) -> MetricsLogger:
        """Run fast headless training loop (no rendering) for benchmark comparisons."""
        for ep in range(1, num_episodes + 1):
            state_vec, snapshot = self.env.reset()
            episode_reward = 0.0
            episode_steps = 0
            episode_losses = []
            done = False

            while not done:
                action = self.agent.select_action(state_vec, eval_mode=False)
                next_state_vec, reward, done, snapshot = self.env.step(action)

                loss = self.agent.learn(state_vec, action, reward, next_state_vec, done)
                if loss > 0.0:
                    episode_losses.append(loss)

                episode_reward += reward
                episode_steps += 1
                state_vec = next_state_vec

            self.agent.update_epsilon()
            metrics = self.logger.log_episode(
                episode=ep,
                score=snapshot.score,
                total_reward=episode_reward,
                steps=episode_steps,
                losses=episode_losses,
                epsilon=self.agent.epsilon
            )

            # Terminal Progress Log
            extra_info = ""
            if hasattr(self.agent, "q_table"):
                extra_info = f" | Q-Table Size: {len(self.agent.q_table)}"
            elif episode_losses:
                extra_info = f" | Avg Loss: {metrics.avg_loss:.4f}"

            print(
                f"[{self.mode_name} Ep {ep:04d}/{num_episodes:04d}] "
                f"Score: {snapshot.score:2d} | "
                f"Reward: {episode_reward:8.1f} | "
                f"Crash Penalty: {RewardConfig.CRASH_PENALTY:6.1f} | "
                f"Steps: {episode_steps:4d} | "
                f"Epsilon: {self.agent.epsilon:.3f} | "
                f"Best: {self.logger.best_score:2d}"
                f"{extra_info}"
            )

        ModelPersistenceManager.save_agent(self.agent, self.mode_name)
        csv_name = "q_learning_metrics.csv" if ("q-learning" in self.mode_name.lower() or self.mode_name.lower() == "qlearning") else "dqn_metrics.csv"
        self.logger.export_csv(os.path.join(LOGS_DIR, csv_name))
        print(f"Metrics saved to logs/{csv_name}")
        return self.logger
