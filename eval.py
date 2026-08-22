"""
Flappy Bird Reinforcement Learning — Standalone Evaluation Entry Point.
Executes trained models in pure evaluation (greedy) mode with trajectory path visualization and seed benchmarking.
"""
import sys
import argparse
from agents.q_learning.q_agent import QLearningAgent
from agents.dqn.dqn_agent import DQNAgent
from evaluation.evaluator import RLEvaluator

def main():
    parser = argparse.ArgumentParser(description="Flappy Bird RL Standalone Evaluation Entry Point")
    parser.add_argument(
        "--mode",
        type=str,
        default="qlearning",
        choices=["qlearning", "dqn"],
        help="RL Algorithm mode: 'qlearning' (Tabular Q-Learning) or 'dqn' (Deep Q-Network)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Optional path to custom trained checkpoint file (.pkl or .pth)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes to execute"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible pipe course layout benchmarking"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run fast evaluation without Pygame GUI display"
    )

    args = parser.parse_args()

    print("==================================================")
    print("   FLAPPY BIRD RL STANDALONE EVALUATION PLATFORM  ")
    print("==================================================")
    print(f"Algorithm Selected : {args.mode.upper()}")
    print(f"Target Episodes    : {args.episodes}")
    print(f"GUI Display        : {not args.headless}")
    print(f"Seed Benchmark     : {args.seed if args.seed is not None else 'Random (Default)'}")
    print("Policy             : Pure Greedy (Epsilon = 0.0)")
    print("Checkpoint Mode    : Read-Only (Safe from overwriting)")
    print("==================================================")

    if args.mode.lower() == "qlearning":
        agent = QLearningAgent()
        mode_name = "Q-Learning"
    else:
        agent = DQNAgent()
        mode_name = "Deep Q-Learning (DQN)"

    evaluator = RLEvaluator(agent=agent, mode_name=mode_name, model_path=args.model_path, seed=args.seed)

    if args.headless:
        print("Starting headless evaluation...")
        logger = evaluator.run_headless_evaluation(num_episodes=args.episodes)
        print(f"Headless evaluation complete! Best score achieved: {logger.best_score}")
    else:
        evaluator.run_gui_evaluation(num_episodes=args.episodes)

if __name__ == "__main__":
    main()
