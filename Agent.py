import numpy as np
import torch
import torch.nn as nn
from numba import cuda, float32

class SwarmRLAgent:
    """Swarm agent with PSO and RL for HFT."""
    def __init__(self, id, state_dim=10, action_dim=4):
        self.id = id
        self.position = np.zeros(4)  # [entry, target, stop, qty]
        self.velocity = np.zeros(4)
        self.pBest = {'position': None, 'fitness': -np.inf}
        # RL Policy Network
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, action_dim), nn.Softmax(dim=-1)
        )
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.q_table = {}  # Q-learning table

    @cuda.jit
    def update_velocity(self, gBest, w=0.8, c1=1.5, c2=1.5):
        """GPU-accelerated PSO velocity update."""
        r1, r2 = np.random.rand(4), np.random.rand(4)
        self.velocity = (w * self.velocity + 
                         c1 * r1 * (self.pBest['position'] - self.position) + 
                         c2 * r2 * (gBest - self.position))

    def propose_trade(self, state):
        """Propose trade using RL policy."""
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action_probs = self.policy_net(state_tensor)
        action = torch.multinomial(action_probs, 1).item()
        return self.map_action_to_trade(action, state)

    def map_action_to_trade(self, action, state):
        """Map RL action to trade parameters."""
        price = state[0]  # Current price from state
        if action == 0:  # Buy small
            return np.array([price, price + 3, price - 2, 1800])
        elif action == 1:  # Buy large
            return np.array([price, price + 3, price - 2, 5400])
        elif action == 2:  # Sell small
            return np.array([price, price - 3, price + 2, 1800])
        else:  # Sell large
            return np.array([price, price - 3, price + 2, 5400])

    def learn(self, state, action, reward, next_state):
        """Hybrid RL: Q-learning + Policy Gradient."""
        state_key = tuple(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(4)
        
        # Q-learning update
        next_q = max(self.q_table.get(tuple(next_state), np.zeros(4)))
        self.q_table[state_key][action] += 0.01 * (reward + 0.95 * next_q - self.q_table[state_key][action])

        # Policy gradient update
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action_probs = self.policy_net(state_tensor)
        loss = -torch.log(action_probs[action]) * reward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def evaluate_fitness(self, trade, market_data):
        """Fitness function: profit - risk - latency."""
        profit = (trade[1] - trade[0]) * trade[3] if trade[1] > trade[0] else (trade[0] - trade[1]) * trade[3]
        risk = abs(trade[0] - trade[2]) * trade[3]  # Potential loss
        latency = 0.001  # Placeholder latency penalty
        return 1.0 * profit - 0.5 * risk - 0.1 * latency
