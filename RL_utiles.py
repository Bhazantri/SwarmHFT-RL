import numpy as np

class RLUtils:
    """Utilities for RL in HFT."""
    @staticmethod
    def discretize_state(state, bins=10):
        """Discretize continuous state for Q-learning."""
        return tuple(np.digitize(state[i], np.linspace(min(state[i], 0), max(state[i], 1), bins)) for i in range(len(state)))

    @staticmethod
    def compute_reward(profit, slippage=0.1, risk_exposure=0.5):
        """Reward function for RL."""
        return profit - 0.1 * slippage - 0.5 * risk_exposure
