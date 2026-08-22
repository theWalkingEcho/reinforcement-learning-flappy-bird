"""
Pygame Game Renderer for Standalone Evaluation Mode.
Renders clean full game view with Score, Best Score, and Motion Trajectory Trails.
"""
import pygame
from collections import deque
from core.entities.game_state import GameState

class EvalGameRenderer:
    # Palette
    COLOR_BG_TOP = (45, 52, 70)       # Dark slate blue
    COLOR_BG_BOTTOM = (25, 30, 42)    # Dark navy
    COLOR_BIRD_BODY = (255, 204, 0)   # Gold / yellow
    COLOR_BIRD_OUTLINE = (230, 150, 0)
    COLOR_BIRD_EYE = (255, 255, 255)
    COLOR_BIRD_PUPIL = (10, 10, 10)
    COLOR_PIPE_BODY = (46, 204, 113)  # Emerald green
    COLOR_PIPE_BORDER = (39, 174, 96)
    COLOR_TEXT = (255, 255, 255)
    COLOR_GOLD = (241, 196, 15)       # Gold accent for best score
    COLOR_TRAIL = (255, 220, 100)     # Glowing trail color

    def __init__(self, surface: pygame.Surface):
        self.surface = surface
        pygame.font.init()
        self.font_score = pygame.font.SysFont("Outfit, Inter, Arial, sans-serif", 38, bold=True)
        self.font_best = pygame.font.SysFont("Outfit, Inter, Arial, sans-serif", 22, bold=True)
        self.font_badge = pygame.font.SysFont("Outfit, Inter, Arial, sans-serif", 14, bold=True)

        # Motion path history (stores recent bird center positions for trajectory rendering)
        self.path_history = deque(maxlen=25)

    def reset_path(self) -> None:
        """Clear historical trajectory trail on episode reset."""
        self.path_history.clear()

    def render(self, state: GameState, best_score: int, mode_name: str = "Q-Learning") -> None:
        """Render frame for current game state with trajectory path visualization."""
        width = self.surface.get_width()
        height = self.surface.get_height()

        # Update motion trail history
        self.path_history.append((int(state.bird.x), int(state.bird.y)))

        # 1. Gradient Background
        for y in range(height):
            ratio = y / float(height)
            r = int(self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio)
            g = int(self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio)
            b = int(self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio)
            pygame.draw.line(self.surface, (r, g, b), (0, y), (width, y))

        # 2. Draw Pipes
        for pipe in state.pipes:
            # Top Pipe
            top_x, top_y, top_r, top_b = pipe.top_pipe_rect
            top_rect = pygame.Rect(int(top_x), 0, int(pipe.width), int(top_b))
            pygame.draw.rect(self.surface, self.COLOR_PIPE_BODY, top_rect)
            pygame.draw.rect(self.surface, self.COLOR_PIPE_BORDER, top_rect, width=3)
            # Pipe rim / cap
            cap_rect_top = pygame.Rect(int(top_x - 4), int(top_b - 20), int(pipe.width + 8), 20)
            pygame.draw.rect(self.surface, self.COLOR_PIPE_BODY, cap_rect_top)
            pygame.draw.rect(self.surface, self.COLOR_PIPE_BORDER, cap_rect_top, width=3)

            # Bottom Pipe
            bot_x, bot_y, bot_r, bot_b = pipe.bottom_pipe_rect
            bot_height = height - int(bot_y)
            bot_rect = pygame.Rect(int(bot_x), int(bot_y), int(pipe.width), int(bot_height))
            pygame.draw.rect(self.surface, self.COLOR_PIPE_BODY, bot_rect)
            pygame.draw.rect(self.surface, self.COLOR_PIPE_BORDER, bot_rect, width=3)
            # Pipe rim / cap
            cap_rect_bot = pygame.Rect(int(bot_x - 4), int(bot_y), int(pipe.width + 8), 20)
            pygame.draw.rect(self.surface, self.COLOR_PIPE_BODY, cap_rect_bot)
            pygame.draw.rect(self.surface, self.COLOR_PIPE_BORDER, cap_rect_bot, width=3)

        # 3. Draw Trajectory Path Ribbon (Glowing trail of best path)
        if len(self.path_history) > 1:
            points = list(self.path_history)
            for i in range(len(points) - 1):
                alpha = int(255 * (i / len(points)))
                radius = max(2, int(8 * (i / len(points))))
                trail_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(trail_surface, (255, 220, 80, alpha // 2), (radius, radius), radius)
                self.surface.blit(trail_surface, (points[i][0] - radius, points[i][1] - radius))

        # 4. Draw Bird
        bird_center = (int(state.bird.x), int(state.bird.y))
        bird_radius = state.bird.radius
        pygame.draw.circle(self.surface, self.COLOR_BIRD_BODY, bird_center, bird_radius)
        pygame.draw.circle(self.surface, self.COLOR_BIRD_OUTLINE, bird_center, bird_radius, width=2)
        # Bird Eye
        eye_center = (int(state.bird.x + bird_radius * 0.4), int(state.bird.y - bird_radius * 0.3))
        pygame.draw.circle(self.surface, self.COLOR_BIRD_EYE, eye_center, 5)
        pygame.draw.circle(self.surface, self.COLOR_BIRD_PUPIL, eye_center, 2)
        # Bird Beak
        beak_pts = [
            (state.bird.x + bird_radius * 0.6, state.bird.y),
            (state.bird.x + bird_radius * 1.2, state.bird.y + 2),
            (state.bird.x + bird_radius * 0.6, state.bird.y + 6)
        ]
        pygame.draw.polygon(self.surface, (255, 100, 0), beak_pts)

        # 5. Clean HUD Overlay (Score & Best Score ONLY)
        hud_surface = pygame.Surface((180, 80), pygame.SRCALPHA)
        hud_surface.fill((15, 22, 35, 190))
        pygame.draw.rect(hud_surface, (255, 255, 255, 40), hud_surface.get_rect(), width=1, border_radius=8)
        self.surface.blit(hud_surface, (15, 15))

        # Render Current Score
        label_score = self.font_badge.render("SCORE", True, (160, 175, 200))
        text_score = self.font_score.render(f"{state.score}", True, self.COLOR_TEXT)
        self.surface.blit(label_score, (25, 20))
        self.surface.blit(text_score, (25, 36))

        # Render Best Score
        label_best = self.font_badge.render("BEST", True, (160, 175, 200))
        text_best = self.font_best.render(f"{best_score}", True, self.COLOR_GOLD)
        self.surface.blit(label_best, (115, 20))
        self.surface.blit(text_best, (115, 45))

        # Evaluation Mode Badge (Top Right)
        badge_surface = self.font_badge.render(f"EVALUATION: {mode_name.upper()}", True, (140, 160, 190))
        self.surface.blit(badge_surface, (width - badge_surface.get_width() - 15, 18))
