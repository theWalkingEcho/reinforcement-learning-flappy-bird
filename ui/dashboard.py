"""
Pygame Live Metrics Dashboard rendering real-time training graphs and statistics panel.
"""
import pygame
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for thread-safe surface rendering
import matplotlib.pyplot as plt
import io
from training.metrics_logger import MetricsLogger

class MetricsDashboard:
    COLOR_BG = (18, 22, 30)
    COLOR_CARD = (28, 35, 48)
    COLOR_TEXT_PRIMARY = (240, 242, 245)
    COLOR_TEXT_SECONDARY = (160, 172, 192)

    def __init__(self, surface: pygame.Surface):
        self.surface = surface
        pygame.font.init()
        self.font_title = pygame.font.SysFont("Outfit, Inter, Arial, sans-serif", 24, bold=True)
        self.font_stat = pygame.font.SysFont("Outfit, Inter, Arial, sans-serif", 20, bold=True)
        self.font_label = pygame.font.SysFont("Outfit, Inter, Arial, sans-serif", 14)

    def render(
        self,
        logger: MetricsLogger,
        current_episode: int,
        agent_name: str,
        q_table_size: int = 0
    ) -> None:
        """Render metrics dashboard cards and real-time Matplotlib plots."""
        width = self.surface.get_width()
        height = self.surface.get_height()

        # Fill background
        self.surface.fill(self.COLOR_BG)

        # Header Title
        title_surf = self.font_title.render("Training Performance Dashboard", True, self.COLOR_TEXT_PRIMARY)
        self.surface.blit(title_surf, (20, 15))

        # Stat Cards (Episode, Best Score, Avg Score (10), Q-Table Size / Loss)
        history = logger.history
        best_score = logger.best_score
        scores = [m.score for m in history]
        recent_avg = float(sum(scores[-10:]) / len(scores[-10:])) if scores else 0.0

        card_y = 50
        card_w = (width - 60) // 3
        card_h = 60

        # Card 1: Total Episodes
        self._draw_card(20, card_y, card_w, card_h, "EPISODE", str(current_episode))
        # Card 2: Highest Score
        self._draw_card(20 + card_w + 10, card_y, card_w, card_h, "BEST SCORE", str(best_score))
        # Card 3: Avg Score (Last 10)
        self._draw_card(20 + (card_w + 10) * 2, card_y, card_w, card_h, "AVG (10 EPS)", f"{recent_avg:.1f}")

        # Render Chart Panel
        chart_surface = self._create_chart_surface(logger, width - 40, height - 130, agent_name)
        if chart_surface:
            self.surface.blit(chart_surface, (20, 120))

    def _draw_card(self, x: int, y: int, w: int, h: int, label: str, value: str) -> None:
        rect = pygame.Rect(x, y, w, h)
        pygame.draw.rect(self.surface, self.COLOR_CARD, rect, border_radius=8)
        pygame.draw.rect(self.surface, (45, 55, 72), rect, width=1, border_radius=8)

        lbl_surf = self.font_label.render(label, True, self.COLOR_TEXT_SECONDARY)
        val_surf = self.font_stat.render(value, True, (46, 204, 113))

        self.surface.blit(lbl_surf, (x + 12, y + 8))
        self.surface.blit(val_surf, (x + 12, y + 28))

    def _create_chart_surface(
        self,
        logger: MetricsLogger,
        w: int,
        h: int,
        agent_name: str
    ) -> pygame.Surface | None:
        """Draw matplotlib plots and convert to Pygame Surface."""
        history = logger.history
        if not history:
            return None

        episodes = [m.episode for m in history]
        scores = [m.score for m in history]
        ma_scores = logger.get_moving_average("score", window=10)
        rewards = [m.total_reward for m in history]
        epsilons = [m.epsilon for m in history]

        # Dark theme style for Matplotlib
        plt.style.use("dark_background")
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(w / 100, h / 100), dpi=100)
        fig.patch.set_facecolor("#12161e")

        # Subplot 1: Score & Moving Avg
        ax1.set_facecolor("#1c2330")
        ax1.plot(episodes, scores, color="#3498db", alpha=0.4, label="Score")
        if ma_scores:
            ax1.plot(episodes, ma_scores, color="#2ecc71", linewidth=2, label="Moving Avg (10)")
        ax1.set_title("Episode Score", fontsize=10, color="#f0f2f5")
        ax1.tick_params(colors="#a0acb0", labelsize=8)
        ax1.grid(True, linestyle="--", alpha=0.2)

        # Subplot 2: Total Reward per Episode
        ax2.set_facecolor("#1c2330")
        ax2.plot(episodes, rewards, color="#e74c3c", linewidth=1.5)
        ax2.set_title("Total Reward", fontsize=10, color="#f0f2f5")
        ax2.tick_params(colors="#a0acb0", labelsize=8)
        ax2.grid(True, linestyle="--", alpha=0.2)

        # Subplot 3: Epsilon Decay
        ax3.set_facecolor("#1c2330")
        ax3.plot(episodes, epsilons, color="#f1c40f", linewidth=1.5)
        ax3.set_title("Epsilon (Exploration)", fontsize=10, color="#f0f2f5")
        ax3.tick_params(colors="#a0acb0", labelsize=8)
        ax3.grid(True, linestyle="--", alpha=0.2)

        # Subplot 4: Loss (if DQN) or Steps survived
        ax4.set_facecolor("#1c2330")
        if "DQN" in agent_name.upper():
            losses = [m.avg_loss for m in history]
            ax4.plot(episodes, losses, color="#9b59b6", linewidth=1.5)
            ax4.set_title("DQN Loss", fontsize=10, color="#f0f2f5")
        else:
            steps = [m.steps for m in history]
            ax4.plot(episodes, steps, color="#9b59b6", linewidth=1.5)
            ax4.set_title("Steps Survived", fontsize=10, color="#f0f2f5")
        ax4.tick_params(colors="#a0acb0", labelsize=8)
        ax4.grid(True, linestyle="--", alpha=0.2)

        plt.tight_layout()

        # Save buffer to Pygame Surface
        buf = io.BytesIO()
        plt.savefig(buf, format="png", facecolor=fig.get_facecolor(), edgecolor="none")
        buf.seek(0)
        plt.close(fig)

        image = pygame.image.load(buf)
        return image
