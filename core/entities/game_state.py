"""
GameState immutable snapshot entity for transferring frame status across layers.
"""
from dataclasses import dataclass
from typing import List
from core.entities.bird import Bird
from core.entities.pipe import PipePair

@dataclass(frozen=True)
class GameState:
    bird: Bird
    pipes: List[PipePair]
    score: int
    is_game_over: bool
    frame_count: int

    @property
    def next_pipe(self) -> PipePair | None:
        """Find the upcoming pipe that the bird has not passed yet."""
        for pipe in self.pipes:
            if pipe.x + pipe.width > self.bird.x - self.bird.radius:
                return pipe
        return self.pipes[0] if self.pipes else None

    @property
    def subsequent_pipe(self) -> PipePair | None:
        """Find the pipe after the next pipe."""
        upcoming = []
        for pipe in self.pipes:
            if pipe.x + pipe.width > self.bird.x - self.bird.radius:
                upcoming.append(pipe)
        return upcoming[1] if len(upcoming) > 1 else (upcoming[0] if upcoming else None)
