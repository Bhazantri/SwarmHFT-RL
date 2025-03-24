import numpy as np
from pandas_ta import trendline

class MicrostructureAnalyzer:
    """Analyzes market microstructure for HFT."""
    def __init__(self):
        self.order_book = {'bids': [], 'asks': []}
        self.price_history = []

    def update(self, tick_data):
        """Update with new tick data."""
        self.price_history.append(tick_data['price'])
        self.order_book['bids'] = tick_data['bids']  # [price, volume]
        self.order_book['asks'] = tick_data['asks']

    def order_flow_imbalance(self):
        """Compute OFI: Delta bids - Delta asks."""
        bid_vol = sum(v for _, v in self.order_book['bids'][-10:])
        ask_vol = sum(v for _, v in self.order_book['asks'][-10:])
        return bid_vol - ask_vol

    def liquidity_shift(self):
        """Compute change in liquidity depth."""
        current_depth = sum(v for _, v in self.order_book['bids']) + sum(v for _, v in self.order_book['asks'])
        return current_depth  # Simplified; ideally compare with previous

    def trendline_slope(self):
        """Dynamic trendline slope."""
        if len(self.price_history) < 10:
            return 0
        prices = np.array(self.price_history[-10:])
        x = np.arange(len(prices))
        slope, _ = np.polyfit(x, prices, 1)
        return slope

    def volatility(self):
        """Short-term volatility."""
        if len(self.price_history) < 10:
            return 0
        return np.std(self.price_history[-10:])

    def get_features(self):
        """Return microstructure features."""
        return {
            'price': self.price_history[-1] if self.price_history else 0,
            'bid_ask_spread': (self.order_book['asks'][0][0] - self.order_book['bids'][0][0]) if self.order_book['asks'] else 0,
            'order_flow_imbalance': self.order_flow_imbalance(),
            'liquidity_shift': self.liquidity_shift(),
            'trendline_slope': self.trendline_slope(),
            'volatility': self.volatility()
        }
