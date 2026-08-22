"""
Pipe Domain Entity representing top and bottom obstacles with dynamic heights.
"""
import random
from dataclasses import dataclass
from utils.config import PhysicsConfig

@dataclass
class PipePair:
    x: float
    gap_center_y: float
    gap_height: float = PhysicsConfig.PIPE_GAP_HEIGHT
    width: float = PhysicsConfig.PIPE_WIDTH
    passed: bool = False

    @classmethod
    def create_random(cls, x_position: float) -> "PipePair":
        """Factory method to create a pipe pair at x_position with a random gap height."""
        gap_center = random.uniform(PhysicsConfig.MIN_PIPE_Y, PhysicsConfig.MAX_PIPE_Y)
        return cls(x=x_position, gap_center_y=gap_center)

    def update(self) -> None:
        """Move pipe leftwards by pipe scroll speed."""
        self.x -= PhysicsConfig.PIPE_SPEED

    @property
    def top_pipe_rect(self) -> tuple[float, float, float, float]:
        """Return (left, top, right, bottom) for top pipe segment."""
        top_bottom_y = self.gap_center_y - self.gap_height / 2.0
        return (self.x, 0.0, self.x + self.width, top_bottom_y)

    @property
    def bottom_pipe_rect(self) -> tuple[float, float, float, float]:
        """Return (left, top, right, bottom) for bottom pipe segment."""
        bottom_top_y = self.gap_center_y + self.gap_height / 2.0
        return (self.x, bottom_top_y, self.x + self.width, PhysicsConfig.MAX_PIPE_Y + 200)

    def collides_with_bird(self, bird_box: tuple[float, float, float, float]) -> bool:
        """Check AABB collision between bird bounding box and top/bottom pipes."""
        bx1, by1, bx2, by2 = bird_box
        
        # Check top pipe
        tx1, ty1, tx2, ty2 = self.top_pipe_rect
        if not (bx2 < tx1 or bx1 > tx2 or by2 < ty1 or by1 > ty2):
            return True

        # Check bottom pipe
        bx1_b, by1_b, bx2_b, by2_b = self.bottom_pipe_rect
        if not (bx2 < bx1_b or bx1 > bx2_b or by2 < by1_b or by1 > by2_b):
            return True

        return False
