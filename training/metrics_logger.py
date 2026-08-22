"""
Metrics Logger for tracking training metrics, history, moving averages, CSV export.
"""
import os
import csv
from dataclasses import dataclass
from typing import List

@dataclass
class EpisodeMetrics:
    episode: int
    score: int
    total_reward: float
    steps: int
    avg_loss: float
    epsilon: float
    best_score: int

class MetricsLogger:
    def __init__(self, agent_name: str = "agent"):
        self.agent_name = agent_name
        self.history: List[EpisodeMetrics] = []
        self.best_score = 0

    def log_episode(
        self,
        episode: int,
        score: int,
        total_reward: float,
        steps: int,
        losses: List[float],
        epsilon: float
    ) -> EpisodeMetrics:
        """Record episode metrics and update best score."""
        if score > self.best_score:
            self.best_score = score

        avg_loss = float(sum(losses) / len(losses)) if losses else 0.0
        metrics = EpisodeMetrics(
            episode=episode,
            score=score,
            total_reward=total_reward,
            steps=steps,
            avg_loss=avg_loss,
            epsilon=epsilon,
            best_score=self.best_score
        )
        self.history.append(metrics)
        return metrics

    def get_moving_average(self, metric: str = "score", window: int = 10) -> List[float]:
        """Compute moving average for a metric series."""
        values = [getattr(m, metric) for m in self.history]
        if not values:
            return []
        ma = []
        for i in range(len(values)):
            start_idx = max(0, i - window + 1)
            ma.append(float(sum(values[start_idx:i+1]) / (i - start_idx + 1)))
        return ma

    def export_csv(self, filepath: str) -> None:
        """Export metrics history to CSV file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        fieldnames = ["episode", "score", "total_reward", "steps", "avg_loss", "epsilon", "best_score"]
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for m in self.history:
                writer.writerow(m.__dict__)
