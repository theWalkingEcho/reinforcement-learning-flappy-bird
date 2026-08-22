"""
Bird Domain Entity representing physics state and collision bounds.
"""
from dataclasses import dataclass
from utils.config import PhysicsConfig

@dataclass
class Bird:
    x: float = PhysicsConfig.BIRD_X
    y: float = PhysicsConfig.BIRD_START_Y
    velocity: float = 0.0
    radius: int = PhysicsConfig.BIRD_RADIUS

    def jump(self) -> None:
        """Apply upward velocity impulse."""
        self.velocity = PhysicsConfig.JUMP_VELOCITY

    def update(self) -> None:
        """Apply gravity and update vertical position."""
        self.velocity += PhysicsConfig.GRAVITY
        if self.velocity > PhysicsConfig.TERMINAL_VELOCITY:
            self.velocity = PhysicsConfig.TERMINAL_VELOCITY
        self.y += self.velocity

    def reset(self, start_y: float = PhysicsConfig.BIRD_START_Y) -> None:
        """Reset bird to initial starting position and velocity."""
        self.x = PhysicsConfig.BIRD_X
        self.y = start_y
        self.velocity = 0.0

    @property
    def bounding_box(self) -> tuple[float, float, float, float]:
        """Return (left, top, right, bottom) bounds."""
        return (
            self.x - self.radius,
            self.y - self.radius,
            self.x + self.radius,
            self.y + self.radius
        )
