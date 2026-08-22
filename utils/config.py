"""
Centralized Configuration for Flappy Bird RL project.
Contains physics, screen dimensions, rewards, state space specs, and hyperparameters.
"""
from dataclasses import dataclass
import os

# Base directory for saving models and logs
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

@dataclass(frozen=True)
class WindowConfig:
    GAME_WIDTH: int = 400
    DASHBOARD_WIDTH: int = 550
    TOTAL_WIDTH: int = 950
    HEIGHT: int = 600
    FPS: int = 60
    TRAIN_FPS: int = 0  # 0 for maximum speed during fast-forward training

@dataclass(frozen=True)
class PhysicsConfig:
    GRAVITY: float = 0.8
    JUMP_VELOCITY: float = -9.0
    TERMINAL_VELOCITY: float = 10.0
    BIRD_X: float = 80.0
    BIRD_START_Y: float = 300.0
    BIRD_RADIUS: int = 15
    PIPE_WIDTH: int = 70
    PIPE_GAP_HEIGHT: int = 140
    PIPE_SPEED: float = 4.0
    PIPE_SPACING: float = 220.0
    MIN_PIPE_Y: int = 100
    MAX_PIPE_Y: int = 380

@dataclass(frozen=True)
class RewardConfig:
    SURVIVAL: float = 0.1
    PIPE_CLEARED: float = 10.0
    CRASH_PENALTY: float = -50.0  # Balanced to avoid massive value distortion in Q-table
    CENTERING_SHAPING: float = 0.05  # Reduced shaping so it doesn't overpower game logic

@dataclass(frozen=True)
class QLearningConfig:
    ALPHA: float = 0.1
    GAMMA: float = 0.99
    EPSILON_START: float = 1.0
    EPSILON_MIN: float = 0.02  # Healthy 2% exploration floor to prevent policy lock
    EPSILON_DECAY: float = 0.999  # Slower decay to explore properly across 5000 episodes
    MODEL_PATH: str = os.path.join(MODELS_DIR, "q_table.pkl")

@dataclass(frozen=True)
class DQNConfig:
    LEARNING_RATE: float = 0.0005
    GAMMA: float = 0.99
    EPSILON_START: float = 1.0
    EPSILON_MIN: float = 0.01
    EPSILON_DECAY: float = 0.996
    BUFFER_CAPACITY: int = 50000
    BATCH_SIZE: int = 64
    TARGET_UPDATE_FREQ: int = 300  # Steps
    HIDDEN_DIM: int = 128
    MODEL_PATH: str = os.path.join(MODELS_DIR, "dqn_model.pth")