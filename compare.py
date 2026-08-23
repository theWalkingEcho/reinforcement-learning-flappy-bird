"""
Head-to-head performance comparison between Q-Learning and Deep Q-Learning (DQN).
"""
import os
import csv
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from training.metrics_logger import EpisodeMetrics, MetricsLogger
from utils.config import LOGS_DIR

def load_metrics(filepath: str, agent_name: str) -> MetricsLogger:
    """Load previously exported training metrics without changing the CSV."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Training metrics not found at {filepath}. "
            "Run training first; comparison will not retrain or create metrics."
        )

    logger = MetricsLogger(agent_name=agent_name)
    with open(filepath, newline="") as metrics_file:
        for row in csv.DictReader(metrics_file):
            logger.history.append(EpisodeMetrics(
                episode=int(row["episode"]),
                score=int(row["score"]),
                total_reward=float(row["total_reward"]),
                steps=int(row["steps"]),
                avg_loss=float(row["avg_loss"]),
                epsilon=float(row["epsilon"]),
                best_score=int(row["best_score"]),
            ))

    if not logger.history:
        raise ValueError(f"Training metrics file is empty: {filepath}")
    logger.best_score = max(metric.best_score for metric in logger.history)
    return logger


def compare_agents() -> None:
    print("==================================================")
    print("      RUNNING HEAD-TO-HEAD ALGORITHM COMPARISON    ")
    print("      Using existing training checkpoints and CSV metrics")
    print("==================================================")

    # 1. Load Tabular Q-Learning metrics
    print("\n--- Loading Q-Learning metrics ---")
    q_logger = load_metrics(os.path.join(LOGS_DIR, "q_learning_metrics.csv"), "Q-Learning")

    # 2. Load Deep Q-Network metrics
    print("\n--- Loading DQN metrics ---")
    dqn_logger = load_metrics(os.path.join(LOGS_DIR, "dqn_metrics.csv"), "DQN")

    # 3. Plot Comparison Charts
    print("\nGenerating comparative plots...")
    q_episodes = [m.episode for m in q_logger.history]
    q_scores = [m.score for m in q_logger.history]
    q_ma = q_logger.get_moving_average("score", window=10)
    q_rewards = [m.total_reward for m in q_logger.history]

    dqn_episodes = [m.episode for m in dqn_logger.history]
    dqn_scores = [m.score for m in dqn_logger.history]
    dqn_ma = dqn_logger.get_moving_average("score", window=10)
    dqn_rewards = [m.total_reward for m in dqn_logger.history]

    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=120)
    fig.patch.set_facecolor("#12161e")

    # Plot 1: Scores & Moving Averages
    ax1.set_facecolor("#1c2330")
    ax1.plot(q_episodes, q_scores, color="#3498db", alpha=0.3, label="Q-Learning Raw")
    ax1.plot(q_episodes, q_ma, color="#2980b9", linewidth=2.5, label="Q-Learning MA(10)")
    ax1.plot(dqn_episodes, dqn_scores, color="#e74c3c", alpha=0.3, label="DQN Raw")
    ax1.plot(dqn_episodes, dqn_ma, color="#c0392b", linewidth=2.5, label="DQN MA(10)")
    ax1.set_title("Episode Score Comparison", fontsize=12, color="#f0f2f5", fontweight="bold")
    ax1.set_xlabel("Episode", fontsize=10, color="#a0acb0")
    ax1.set_ylabel("Pipes Cleared (Score)", fontsize=10, color="#a0acb0")
    ax1.tick_params(colors="#a0acb0")
    ax1.grid(True, linestyle="--", alpha=0.2)
    ax1.legend(loc="upper left")

    # Plot 2: Total Rewards
    ax2.set_facecolor("#1c2330")
    q_reward_ma = q_logger.get_moving_average("total_reward", window=10)
    dqn_reward_ma = dqn_logger.get_moving_average("total_reward", window=10)
    ax2.plot(q_episodes, q_reward_ma, color="#2ecc71", linewidth=2, label="Q-Learning Rewards MA(10)")
    ax2.plot(dqn_episodes, dqn_reward_ma, color="#f39c12", linewidth=2, label="DQN Rewards MA(10)")
    ax2.set_title("Total Cumulative Reward Comparison", fontsize=12, color="#f0f2f5", fontweight="bold")
    ax2.set_xlabel("Episode", fontsize=10, color="#a0acb0")
    ax2.set_ylabel("Total Episode Reward", fontsize=10, color="#a0acb0")
    ax2.tick_params(colors="#a0acb0")
    ax2.grid(True, linestyle="--", alpha=0.2)
    ax2.legend(loc="upper left")

    plt.tight_layout()
    plot_path = os.path.join(LOGS_DIR, "q_vs_dqn_comparison.png")
    plt.savefig(plot_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)

    print("==================================================")
    print("               COMPARISON RESULTS                 ")
    print("==================================================")
    print(f"Q-Learning Max Score : {q_logger.best_score}")
    print(f"DQN Max Score        : {dqn_logger.best_score}")
    print(f"Metrics CSV Saved to : {LOGS_DIR}")
    print(f"Comparison Plot Saved: {plot_path}")
    print("==================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Q-Learning vs DQN")
    args = parser.parse_args()
    compare_agents()
