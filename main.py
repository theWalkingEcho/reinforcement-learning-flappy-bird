"""
Flappy Bird Reinforcement Learning — Main Entry Point.
Allows selecting algorithm (Q-Learning / DQN) and running interactive training GUI.
"""
import sys
import argparse
from agents.q_learning.q_agent import QLearningAgent
from agents.dqn.dqn_agent import DQNAgent
from training.trainer import RLTrainer

def main():
    parser = argparse.ArgumentParser(description="Flappy Bird Reinforcement Learning Platform")
    parser.add_argument(
        "--mode",
        type=str,
        default="qlearning",
        choices=["qlearning", "dqn"],
        help="RL Algorithm mode: 'qlearning' (Tabular) or 'dqn' (Deep Q-Network)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Maximum number of training episodes"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run fast headless training without GUI display"
    )

    args = parser.parse_args()

    print("==================================================")
    print("   FLAPPY BIRD REINFORCEMENT LEARNING PLATFORM    ")
    print("==================================================")
    print(f"Algorithm Selected : {args.mode.upper()}")
    print(f"Target Episodes    : {args.episodes}")
    print(f"GUI Mode Enabled   : {not args.headless}")
    print("--------------------------------------------------")
    print("Controls during GUI training:")
    print("  [SPACE] - Toggle Speed Boost (Fast Training)")
    print("  [P]     - Pause / Resume Training")
    print("  [S]     - Save Checkpoint manually")
    print("==================================================")

    # Initialize Agent based on mode
    if args.mode.lower() == "qlearning":
        agent = QLearningAgent()
        mode_name = "Q-Learning"
    else:
        agent = DQNAgent()
        mode_name = "Deep Q-Learning (DQN)"

    trainer = RLTrainer(agent=agent, mode_name=mode_name)

    if args.headless:
        print("Starting headless training...")
        logger = trainer.run_headless_training(num_episodes=args.episodes)
        print(f"Headless training complete! Best score achieved: {logger.best_score}")
    else:
        trainer.run_gui_training(num_episodes=args.episodes)

if __name__ == "__main__":
    main()
