"""
Pygame Game Renderer for drawing Flappy Bird viewport.
"""
import pygame
from core.entities.game_state import GameState
from utils.config import WindowConfig, PhysicsConfig

class GameRenderer:
    # Palette
    COLOR_BG_TOP = (45, 52, 70)       # Dark slate blue
    COLOR_BG_BOTTOM = (25, 30, 42)    # Dark navy
    COLOR_BIRD_BODY = (255, 204, 0)   # Vibrant gold / yellow
    COLOR_BIRD_OUTLINE = (230, 150, 0)
    COLOR_BIRD_EYE = (255, 255, 255)
    COLOR_BIRD_PUPIL = (10, 10, 10)
    COLOR_PIPE_BODY = (46, 204, 113)  # Vibrant emerald green
    COLOR_PIPE_BORDER = (39, 174, 96)
    COLOR_TEXT = (240, 240, 240)
    COLOR_HUD_BG = (0, 0, 0, 120)

    def __init__(self, surface: pygame.Surface):
        self.surface = surface
        pygame.font.init()
        self.font_large = pygame.font.SysFont("Outfit, Inter, Arial, sans-serif", 32, bold=True)
        self.font_small = pygame.font.SysFont("Outfit, Inter, Arial, sans-serif", 18, bold=True)

    def render(self, state: GameState, agent_name: str = "Agent", epsilon: float = 1.0) -> None:
        """Render frame for current game state into the assigned surface."""
        width = self.surface.get_width()
        height = self.surface.get_height()

        # 1. Draw gradient background
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

        # 3. Draw Bird
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

        # 4. HUD Overlays
        # Mode Badge
        badge_text = self.font_small.render(f"Mode: {agent_name} | ε: {epsilon:.3f}", True, self.COLOR_TEXT)
        self.surface.blit(badge_text, (15, 15))

        # Score
        score_text = self.font_large.render(f"Score: {state.score}", True, (255, 255, 255))
        self.surface.blit(score_text, (15, 45))
