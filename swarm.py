import numpy as np
from src.agent import SwarmRLAgent

class SwarmHFT:
    """Orchestrates swarm of agents for HFT."""
    def __init__(self, num_agents=100, max_quantity=5400):
        self.agents = [SwarmRLAgent(i) for i in range(num_agents)]
        self.gBest = {'position': None, 'fitness': -np.inf}
        self.max_quantity = max_quantity

    def run(self, market_data):
        """Run one iteration of swarm trading."""
        state = self.extract_state(market_data)
        trades = []
        for agent in self.agents:
            trade = agent.propose_trade(state)
            fitness = agent.evaluate_fitness(trade, market_data)
            if fitness > agent.pBest['fitness']:
                agent.pBest = {'position': trade.copy(), 'fitness': fitness}
            if fitness > self.gBest['fitness']:
                self.gBest = {'position': trade.copy(), 'fitness': fitness}
            agent.update_velocity(self.gBest['position'])
            trades.append((trade, fitness))
            agent.learn(state, np.argmax(trade), fitness, state)  # Simplified next_state
        return self.select_best_trade(trades)

    def extract_state(self, market_data):
        """Extract state from market data."""
        return np.array([
            market_data['price'], market_data['bid_ask_spread'], 
            market_data['order_flow_imbalance'], market_data['liquidity_shift'],
            market_data['trendline_slope'], market_data['volatility'],
            0, 0, 0, 0  # Padding to state_dim=10
        ])

    def select_best_trade(self, trades):
        """Consensus: Select top trade by fitness."""
        trades.sort(key=lambda x: x[1], reverse=True)
        return trades[0][0] if trades else None
