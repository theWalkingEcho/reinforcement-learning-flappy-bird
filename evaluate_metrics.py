"""
RL Evaluation Metrics Report - Flappy Bird
==========================================
Loads trained model checkpoints, runs live headless evaluation episodes,
computes all standard RL evaluation metrics, and generates an 8-panel report.

Metrics covered:
  1. Cumulative Reward Curve
  2. Average Reward per Step (reward efficiency)
  3. Discounted Reward (gamma=0.99)
  4. Convergence Rate (rolling mean vs threshold)
  5. Score Distribution (violin plot)
  6. Stability - Rolling Std Dev of Score
  7. Sample Efficiency (steps to score thresholds)
  8. Summary Statistics Table

Usage:
  python evaluate_metrics.py
  python evaluate_metrics.py --episodes 200 --gamma 0.95 --threshold 5
  python evaluate_metrics.py --episodes 100 --seed 42
"""
import os
import sys
import math
import random
import argparse
from typing import List, Optional
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# Add project root to path so internal modules resolve correctly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.q_learning.q_agent import QLearningAgent
from agents.dqn.dqn_agent import DQNAgent
from environment.flappy_env import FlappyBirdEnv
from utils.config import QLearningConfig, DQNConfig, LOGS_DIR

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
NUM_EPISODES  = 100           # default eval episodes per agent
GAMMA         = 0.99          # discount factor for discounted reward metric
CONV_THRESHOLD = 5            # pipes: MA(10) >= this => converged
MA_WINDOW     = 10
OUT_PATH      = os.path.join(LOGS_DIR, "rl_evaluation_report.png")

# Palette (dark theme)
BG_DARK   = "#0d1117"
BG_PANEL  = "#161b22"
BG_PANEL2 = "#1c2330"
C_Q       = "#58a6ff"
C_DQN     = "#f78166"
C_TEXT    = "#c9d1d9"
C_MUTED   = "#8b949e"
C_GRID    = "#30363d"


# ─────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────
@dataclass
class EpisodeResult:
    episode:      int
    score:        int
    total_reward: float
    steps:        int


# ─────────────────────────────────────────────
# Live Headless Evaluation Runner
# ─────────────────────────────────────────────
def run_evaluation(agent, num_episodes: int, seed: Optional[int], label: str) -> List[EpisodeResult]:
    """
    Load the trained checkpoint, run greedy (eval_mode=True) episodes
    against the live environment, and return per-episode results.
    """
    env = FlappyBirdEnv()
    results = []

    for ep in range(1, num_episodes + 1):
        if seed is not None:
            random.seed(seed + ep)

        state_vec, snapshot = env.reset()
        episode_reward = 0.0
        episode_steps  = 0
        done = False

        while not done:
            action = agent.select_action(state_vec, eval_mode=True)
            next_state_vec, reward, done, snapshot = env.step(action)
            episode_reward += reward
            episode_steps  += 1
            state_vec = next_state_vec

        results.append(EpisodeResult(
            episode      = ep,
            score        = snapshot.score,
            total_reward = episode_reward,
            steps        = episode_steps,
        ))

        extra = f" | Q-Table size: {len(agent.q_table)}" if hasattr(agent, "q_table") else ""
        print(
            f"  [{label}] Ep {ep:04d}/{num_episodes:04d} | "
            f"Score: {snapshot.score:3d} | "
            f"Reward: {episode_reward:8.1f} | "
            f"Steps: {episode_steps:5d}"
            f"{extra}"
        )

    return results


# ─────────────────────────────────────────────
# Metric Calculations
# ─────────────────────────────────────────────
def moving_average(values: List[float], window: int) -> List[float]:
    ma = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        ma.append(float(sum(values[start:i+1]) / (i - start + 1)))
    return ma


def cumulative_reward(results: List[EpisodeResult]) -> List[float]:
    cum, total = [], 0.0
    for r in results:
        total += r.total_reward
        cum.append(total)
    return cum


def avg_reward_per_step(results: List[EpisodeResult]) -> List[float]:
    return [r.total_reward / max(r.steps, 1) for r in results]


def discounted_reward(results: List[EpisodeResult], gamma: float) -> List[float]:
    """
    Episode-level discounted reward approximation.
    Treats total_reward as uniformly distributed over T steps,
    then discounts: disc_R = (R/T) * (1 - gamma^T) / (1 - gamma)
    """
    out = []
    for r in results:
        T = max(r.steps, 1)
        per_step = r.total_reward / T
        if abs(gamma - 1.0) < 1e-9:
            disc = per_step * T
        else:
            disc = per_step * (1.0 - gamma ** T) / (1.0 - gamma)
        out.append(disc)
    return out


def convergence_episode(results: List[EpisodeResult], threshold: int, window: int) -> Optional[int]:
    scores = [r.score for r in results]
    ma = moving_average(scores, window)
    for i, v in enumerate(ma):
        if v >= threshold:
            return results[i].episode
    return None


def rolling_std(values: List[float], window: int) -> List[float]:
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        seg  = values[start:i+1]
        mean = sum(seg) / len(seg)
        var  = sum((x - mean) ** 2 for x in seg) / len(seg)
        result.append(math.sqrt(var))
    return result


def steps_to_threshold(results: List[EpisodeResult], thresholds: List[int]) -> dict:
    cum_steps = 0
    out = {t: None for t in thresholds}
    remaining = set(thresholds)
    for r in results:
        cum_steps += r.steps
        for t in list(remaining):
            if r.score >= t:
                out[t] = cum_steps
                remaining.discard(t)
        if not remaining:
            break
    return out


def summary_stats(results: List[EpisodeResult]) -> dict:
    scores  = [r.score        for r in results]
    rewards = [r.total_reward for r in results]
    steps   = [r.steps        for r in results]
    n       = len(scores)

    def pct(lst, p):
        s   = sorted(lst)
        idx = int(p / 100 * (n - 1))
        return s[idx]

    mean_s = sum(scores) / n
    std_s  = math.sqrt(sum((s - mean_s) ** 2 for s in scores) / n)
    mean_r = sum(rewards) / n
    std_r  = math.sqrt(sum((r - mean_r) ** 2 for r in rewards) / n)

    return {
        "Episodes":      n,
        "Mean Score":    round(mean_s,  2),
        "Std Score":     round(std_s,   2),
        "Max Score":     max(scores),
        "Median Score":  pct(scores,    50),
        "P75 Score":     pct(scores,    75),
        "P90 Score":     pct(scores,    90),
        "Mean Reward":   round(mean_r,  2),
        "Std Reward":    round(std_r,   2),
        "Mean Steps":    round(sum(steps) / n, 1),
        "Total Steps":   sum(steps),
    }


# ─────────────────────────────────────────────
# Plot Helpers
# ─────────────────────────────────────────────
def style_ax(ax, title: str, xlabel: str = "Episode", ylabel: str = ""):
    ax.set_facecolor(BG_PANEL2)
    ax.set_title(title, fontsize=9.5, color=C_TEXT, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=8, color=C_MUTED)
    ax.set_ylabel(ylabel, fontsize=8, color=C_MUTED)
    ax.tick_params(colors=C_MUTED, labelsize=7)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color(C_GRID)
    ax.grid(True, linestyle="--", alpha=0.25, color=C_GRID)


def add_conv_vline(ax, ep: Optional[int], color: str, label: str):
    if ep is not None:
        ax.axvline(ep, color=color, linestyle=":", linewidth=1.4, alpha=0.85,
                   label=f"Converged Ep {ep} ({label})")
    else:
        ax.plot([], [], color=color, linestyle=":", linewidth=1.4,
                label=f"Not converged ({label})")


# ─────────────────────────────────────────────
# Report Builder
# ─────────────────────────────────────────────
def build_report(
    q_results:   List[EpisodeResult],
    dqn_results: List[EpisodeResult],
    gamma:       float,
    threshold:   int,
    out_path:    str,
):
    ep_q   = [r.episode for r in q_results]
    ep_dqn = [r.episode for r in dqn_results]

    q_scores   = [r.score for r in q_results]
    dqn_scores = [r.score for r in dqn_results]

    q_ma   = moving_average(q_scores,   MA_WINDOW)
    dqn_ma = moving_average(dqn_scores, MA_WINDOW)

    q_cum       = cumulative_reward(q_results)
    dqn_cum     = cumulative_reward(dqn_results)

    q_eff       = avg_reward_per_step(q_results)
    dqn_eff     = avg_reward_per_step(dqn_results)
    q_eff_ma    = moving_average(q_eff,   MA_WINDOW)
    dqn_eff_ma  = moving_average(dqn_eff, MA_WINDOW)

    q_disc      = discounted_reward(q_results,   gamma)
    dqn_disc    = discounted_reward(dqn_results, gamma)
    q_disc_ma   = moving_average(q_disc,   MA_WINDOW)
    dqn_disc_ma = moving_average(dqn_disc, MA_WINDOW)

    q_conv   = convergence_episode(q_results,   threshold, MA_WINDOW)
    dqn_conv = convergence_episode(dqn_results, threshold, MA_WINDOW)

    q_std    = rolling_std(q_scores,   MA_WINDOW)
    dqn_std  = rolling_std(dqn_scores, MA_WINDOW)

    thresholds     = [1, 5, 10, 20, 30, 50]
    q_se     = steps_to_threshold(q_results,   thresholds)
    dqn_se   = steps_to_threshold(dqn_results, thresholds)

    q_stats   = summary_stats(q_results)
    dqn_stats = summary_stats(dqn_results)

    # ── Figure ───────────────────────────────────
    fig = plt.figure(figsize=(20, 24), dpi=120, facecolor=BG_DARK)
    gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.35,
                            left=0.06, right=0.97, top=0.95, bottom=0.03)

    fig.text(0.5, 0.975, "Reinforcement Learning Evaluation Report - Flappy Bird",
             ha="center", va="top", fontsize=16, fontweight="bold", color=C_TEXT,
             fontfamily="DejaVu Sans")
    fig.text(0.5, 0.962,
             f"Q-Learning vs DQN  |  Greedy policy on trained checkpoints  |  "
             f"gamma={gamma}  |  Convergence: MA({MA_WINDOW}) >= {threshold} pipes  |  "
             f"{len(q_results)} evaluation episodes each",
             ha="center", va="top", fontsize=8.5, color=C_MUTED)

    # ── [1] Cumulative Reward ────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(ep_q,   q_cum,   color=C_Q,   linewidth=2,   label="Q-Learning")
    ax1.plot(ep_dqn, dqn_cum, color=C_DQN, linewidth=2,   label="DQN")
    ax1.fill_between(ep_q,   q_cum,   alpha=0.12, color=C_Q)
    ax1.fill_between(ep_dqn, dqn_cum, alpha=0.12, color=C_DQN)
    style_ax(ax1, "[1] Cumulative Reward", ylabel="Total Reward Accumulated")
    ax1.legend(fontsize=7.5, facecolor=BG_PANEL, edgecolor=C_GRID, labelcolor=C_TEXT)

    # ── [2] Avg Reward per Step ──────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(ep_q,   q_eff,      color=C_Q,   alpha=0.2,  linewidth=1)
    ax2.plot(ep_dqn, dqn_eff,    color=C_DQN, alpha=0.2,  linewidth=1)
    ax2.plot(ep_q,   q_eff_ma,   color=C_Q,   linewidth=2.2, label=f"Q-Learning MA({MA_WINDOW})")
    ax2.plot(ep_dqn, dqn_eff_ma, color=C_DQN, linewidth=2.2, label=f"DQN MA({MA_WINDOW})")
    style_ax(ax2, "[2] Avg Reward per Step  (Reward Efficiency)", ylabel="Reward / Step")
    ax2.legend(fontsize=7.5, facecolor=BG_PANEL, edgecolor=C_GRID, labelcolor=C_TEXT)

    # ── [3] Discounted Reward ────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(ep_q,   q_disc,      color=C_Q,   alpha=0.2,  linewidth=1)
    ax3.plot(ep_dqn, dqn_disc,    color=C_DQN, alpha=0.2,  linewidth=1)
    ax3.plot(ep_q,   q_disc_ma,   color=C_Q,   linewidth=2.2, label=f"Q-Learning MA({MA_WINDOW})")
    ax3.plot(ep_dqn, dqn_disc_ma, color=C_DQN, linewidth=2.2, label=f"DQN MA({MA_WINDOW})")
    style_ax(ax3, f"[3] Discounted Reward  (gamma={gamma})", ylabel="Discounted Episode Reward")
    ax3.legend(fontsize=7.5, facecolor=BG_PANEL, edgecolor=C_GRID, labelcolor=C_TEXT)

    # ── [4] Convergence Rate ─────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(ep_q,   q_scores,   color=C_Q,   alpha=0.15, linewidth=1)
    ax4.plot(ep_dqn, dqn_scores, color=C_DQN, alpha=0.15, linewidth=1)
    ax4.plot(ep_q,   q_ma,   color=C_Q,   linewidth=2.2, label=f"Q-Learning MA({MA_WINDOW})")
    ax4.plot(ep_dqn, dqn_ma, color=C_DQN, linewidth=2.2, label=f"DQN MA({MA_WINDOW})")
    ax4.axhline(threshold, color="#f0e68c", linestyle="--", linewidth=1.2, alpha=0.75,
                label=f"Threshold = {threshold} pipes")
    add_conv_vline(ax4, q_conv,   C_Q,   "Q-Learning")
    add_conv_vline(ax4, dqn_conv, C_DQN, "DQN")
    style_ax(ax4, "[4] Convergence Rate", ylabel="Score (Pipes Cleared)")
    ax4.legend(fontsize=7, facecolor=BG_PANEL, edgecolor=C_GRID, labelcolor=C_TEXT)

    # ── [5] Score Distribution ───────────────────
    ax5 = fig.add_subplot(gs[2, 0])
    vp = ax5.violinplot(
        [q_scores, dqn_scores],
        positions=[1, 2],
        showmedians=True,
        showextrema=True,
    )
    for i, body in enumerate(vp["bodies"]):
        body.set_facecolor([C_Q, C_DQN][i])
        body.set_alpha(0.45)
    for part in ["cmedians", "cmins", "cmaxes", "cbars"]:
        vp[part].set_color(C_MUTED)
        vp[part].set_linewidth(1.3)
    ax5.set_xticks([1, 2])
    ax5.set_xticklabels(["Q-Learning", "DQN"], color=C_TEXT, fontsize=8.5)
    style_ax(ax5, "[5] Score Distribution  (Violin Plot)",
             xlabel="Algorithm", ylabel="Score (Pipes Cleared)")

    # ── [6] Stability - Rolling Std Dev ─────────
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(ep_q,   q_std,   color=C_Q,   linewidth=2, label="Q-Learning")
    ax6.plot(ep_dqn, dqn_std, color=C_DQN, linewidth=2, label="DQN")
    ax6.fill_between(ep_q,   q_std,   alpha=0.12, color=C_Q)
    ax6.fill_between(ep_dqn, dqn_std, alpha=0.12, color=C_DQN)
    style_ax(ax6, f"[6] Stability - Rolling Std Dev of Score  (window={MA_WINDOW})",
             ylabel="Std Dev of Score")
    ax6.legend(fontsize=7.5, facecolor=BG_PANEL, edgecolor=C_GRID, labelcolor=C_TEXT)
    ax6.text(0.02, 0.92, "Lower = More Stable", transform=ax6.transAxes,
             fontsize=7, color=C_MUTED, ha="left")

    # ── [7] Sample Efficiency ────────────────────
    ax7 = fig.add_subplot(gs[3, 0])
    x     = np.arange(len(thresholds))
    bar_w = 0.35
    q_vals   = [q_se[t]   if q_se[t]   else 0 for t in thresholds]
    dqn_vals = [dqn_se[t] if dqn_se[t] else 0 for t in thresholds]

    bars_q   = ax7.bar(x - bar_w/2, q_vals,   bar_w, color=C_Q,   alpha=0.82, label="Q-Learning")
    bars_dqn = ax7.bar(x + bar_w/2, dqn_vals, bar_w, color=C_DQN, alpha=0.82, label="DQN")

    for bar, t in zip(bars_q, thresholds):
        lbl = f"{q_se[t]:,}" if q_se[t] else "N/A"
        ypos = bar.get_height() + max(q_vals + dqn_vals) * 0.02 if bar.get_height() > 0 else max(q_vals + dqn_vals) * 0.03
        ax7.text(bar.get_x() + bar.get_width()/2, ypos, lbl,
                 ha="center", va="bottom", fontsize=6.5, color=C_Q if q_se[t] else C_MUTED)

    for bar, t in zip(bars_dqn, thresholds):
        lbl = f"{dqn_se[t]:,}" if dqn_se[t] else "N/A"
        ypos = bar.get_height() + max(q_vals + dqn_vals) * 0.02 if bar.get_height() > 0 else max(q_vals + dqn_vals) * 0.03
        ax7.text(bar.get_x() + bar.get_width()/2, ypos, lbl,
                 ha="center", va="bottom", fontsize=6.5, color=C_DQN if dqn_se[t] else C_MUTED)

    ax7.set_xticks(x)
    ax7.set_xticklabels([f"Score>={t}" for t in thresholds], fontsize=7.5, color=C_MUTED)
    style_ax(ax7, "[7] Sample Efficiency  (Cumulative Steps to Reach Score Threshold)",
             xlabel="Score Threshold", ylabel="Cumulative Env Steps")
    ax7.legend(fontsize=7.5, facecolor=BG_PANEL, edgecolor=C_GRID, labelcolor=C_TEXT)
    ax7.text(0.02, 0.92, "Lower = More Sample Efficient", transform=ax7.transAxes,
             fontsize=7, color=C_MUTED)

    # ── [8] Summary Statistics Table ────────────
    ax8 = fig.add_subplot(gs[3, 1])
    ax8.set_facecolor(BG_PANEL2)
    ax8.axis("off")
    ax8.set_title("[8] Summary Statistics", fontsize=9.5, color=C_TEXT,
                  fontweight="bold", pad=8, loc="center")

    stat_keys  = list(q_stats.keys())
    table_data = []
    for k in stat_keys:
        qv = q_stats[k];   dv = dqn_stats[k]
        qf = f"{qv:,}" if isinstance(qv, int) else str(qv)
        df = f"{dv:,}" if isinstance(dv, int) else str(dv)
        table_data.append([k, qf, df])

    tbl = ax8.table(
        cellText  = table_data,
        colLabels = ["Metric", "Q-Learning", "DQN"],
        cellLoc   = "center",
        loc       = "center",
        bbox      = [0.0, 0.0, 1.0, 1.0],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor(C_GRID)
        cell.set_linewidth(0.6)
        if row == 0:
            colors = [BG_DARK, C_Q, C_DQN]
            cell.set_facecolor(colors[col])
            cell.set_text_props(color=C_TEXT if col == 0 else "#ffffff", fontweight="bold")
        else:
            cell.set_facecolor(BG_PANEL if row % 2 == 0 else BG_PANEL2)
            cell.set_text_props(color=[C_MUTED, C_Q, C_DQN][col])

    # ── Footer ───────────────────────────────────
    q_conv_str   = f"Ep {q_conv}"   if q_conv   else "Not reached"
    dqn_conv_str = f"Ep {dqn_conv}" if dqn_conv else "Not reached"
    footer = (
        f"Convergence (MA >= {threshold}):  Q-Learning -> {q_conv_str}   |   DQN -> {dqn_conv_str}   "
        f"|   Total Eval Steps:  Q-Learning -> {q_stats['Total Steps']:,}   |   DQN -> {dqn_stats['Total Steps']:,}"
    )
    fig.text(0.5, 0.012, footer, ha="center", fontsize=8, color=C_MUTED,
             bbox=dict(facecolor=BG_PANEL, edgecolor=C_GRID, boxstyle="round,pad=0.3"))

    os.makedirs(LOGS_DIR, exist_ok=True)
    plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    print(f"[REPORT] Saved -> {out_path}")


# ─────────────────────────────────────────────
# Terminal Summary
# ─────────────────────────────────────────────
def print_summary(q_results, dqn_results, gamma, threshold):
    q_stats   = summary_stats(q_results)
    dqn_stats = summary_stats(dqn_results)
    q_conv    = convergence_episode(q_results,   threshold, MA_WINDOW)
    dqn_conv  = convergence_episode(dqn_results, threshold, MA_WINDOW)
    thresholds = [1, 5, 10, 20, 30, 50]
    q_se      = steps_to_threshold(q_results,   thresholds)
    dqn_se    = steps_to_threshold(dqn_results, thresholds)

    print()
    print("=" * 62)
    print("  EVALUATION METRIC SUMMARY")
    print("=" * 62)
    print(f"  {'Metric':<28} {'Q-Learning':>14} {'DQN':>14}")
    print("  " + "-" * 58)
    for k in q_stats:
        qv = q_stats[k];  dv = dqn_stats[k]
        qf = f"{qv:,}" if isinstance(qv, int) else str(qv)
        df = f"{dv:,}" if isinstance(dv, int) else str(dv)
        print(f"  {k:<28} {qf:>14} {df:>14}")
    print()
    qcs = f"Ep {q_conv}"   if q_conv   else "Not reached"
    dcs = f"Ep {dqn_conv}" if dqn_conv else "Not reached"
    print(f"  {'Convergence Ep':<28} {qcs:>14} {dcs:>14}")
    print()
    print("  Sample Efficiency (cumulative steps to score threshold):")
    for t in thresholds:
        qs = f"{q_se[t]:,}"   if q_se[t]   else "N/A"
        ds = f"{dqn_se[t]:,}" if dqn_se[t] else "N/A"
        print(f"    Score >= {t:<4}: Q-Learning = {qs:>10}   DQN = {ds:>10}")
    print("=" * 62)


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="RL Evaluation Metrics Report - runs trained checkpoints live"
    )
    parser.add_argument("--episodes",  type=int,   default=NUM_EPISODES,
                        help=f"Evaluation episodes per agent (default: {NUM_EPISODES})")
    parser.add_argument("--gamma",     type=float, default=GAMMA,
                        help=f"Discount factor for discounted reward metric (default: {GAMMA})")
    parser.add_argument("--threshold", type=int,   default=CONV_THRESHOLD,
                        help=f"Convergence score threshold MA({MA_WINDOW}) (default: {CONV_THRESHOLD})")
    parser.add_argument("--seed",      type=int,   default=None,
                        help="Random seed for reproducible pipe layouts (default: random)")
    parser.add_argument("--q-model",   type=str,   default=QLearningConfig.MODEL_PATH,
                        help="Path to Q-Learning checkpoint (.pkl)")
    parser.add_argument("--dqn-model", type=str,   default=DQNConfig.MODEL_PATH,
                        help="Path to DQN checkpoint (.pth)")
    parser.add_argument("--out",       type=str,   default=OUT_PATH,
                        help="Output PNG path for the report")
    args = parser.parse_args()

    print("=" * 62)
    print("    RL EVALUATION METRICS REPORT - FLAPPY BIRD")
    print("=" * 62)
    print(f"  Episodes per agent : {args.episodes}")
    print(f"  Discount gamma     : {args.gamma}")
    print(f"  Conv. threshold    : MA({MA_WINDOW}) >= {args.threshold} pipes")
    print(f"  Seed               : {args.seed if args.seed is not None else 'random'}")
    print(f"  Q-Learning model   : {args.q_model}")
    print(f"  DQN model          : {args.dqn_model}")
    print(f"  Output             : {args.out}")
    print("=" * 62)

    # ── Load Q-Learning agent ────────────────────
    print("\n[1/2] Loading Q-Learning agent...")
    q_agent = QLearningAgent()
    if os.path.exists(args.q_model):
        q_agent.load(args.q_model)
        print(f"      Checkpoint loaded: {args.q_model}")
        if hasattr(q_agent, "q_table"):
            print(f"      Q-Table size: {len(q_agent.q_table)} states")
    else:
        print(f"      WARNING: No checkpoint at {args.q_model} - running with random policy")

    print(f"\n      Running {args.episodes} greedy evaluation episodes...")
    q_results = run_evaluation(q_agent, args.episodes, args.seed, "Q-Learning")

    # ── Load DQN agent ───────────────────────────
    print("\n[2/2] Loading DQN agent...")
    dqn_agent = DQNAgent()
    if os.path.exists(args.dqn_model):
        dqn_agent.load(args.dqn_model)
        print(f"      Checkpoint loaded: {args.dqn_model}")
    else:
        print(f"      WARNING: No checkpoint at {args.dqn_model} - running with random policy")

    print(f"\n      Running {args.episodes} greedy evaluation episodes...")
    dqn_results = run_evaluation(dqn_agent, args.episodes, args.seed, "DQN")

    # ── Generate report ──────────────────────────
    print("\n[3/3] Generating evaluation report...")
    build_report(q_results, dqn_results,
                 gamma=args.gamma, threshold=args.threshold, out_path=args.out)

    print_summary(q_results, dqn_results, args.gamma, args.threshold)
    print("\n  Done.")


if __name__ == "__main__":
    main()
