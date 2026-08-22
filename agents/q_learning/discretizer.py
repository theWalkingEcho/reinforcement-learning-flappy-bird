"""
State Discretizer for Tabular Q-Learning.
Maps continuous environment features into discrete state tuples.
"""
from typing import Dict, Tuple
import numpy as np

class StateDiscretizer:
    def __init__(self):
        # Define bin edges for each feature
        # y_diff range ~ [-350, 350]
        self.y_diff_bins = np.linspace(-300, 300, num=16)
        # x_diff range ~ [0, 450]
        self.x_diff_bins = np.linspace(0, 420, num=14)
        # velocity range ~ [-9, 10]
        self.velocity_bins = np.linspace(-9.0, 10.0, num=10)

    def discretize(self, state_dict: Dict[str, float]) -> Tuple[int, int, int]:
        """Map raw state vector dictionary to discrete (y_bin, x_bin, v_bin) tuple."""
        y_bin = int(np.digitize(state_dict["y_diff"], self.y_diff_bins))
        x_bin = int(np.digitize(state_dict["x_diff"], self.x_diff_bins))
        v_bin = int(np.digitize(state_dict["velocity"], self.velocity_bins))
        return (y_bin, x_bin, v_bin)
