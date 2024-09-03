"""
strategy.py

This file contains abstract class definitions that are designed to be compatible with Hindsight's internal
simulation environment. The `Strategy` class provides a base structure that can be inherited and implemented
by user-defined strategies.

### Overview:
Each strategy is defined as a subclass of the `Strategy` class. A strategy represents a set of trading
decisions or actions that are executed after observing the available market data characteristics at each time
step, typically each day. The class's design focuses on the sequential nature of trading, where decisions
are made based on the information available up to the current point in time.

### Core Design:
The core of the Hindsight strategy simulator operates by advancing a single pointer through the simulation's
environment, which keeps track of the current day (or time step) during the simulation, up until the final time
`T`. The `next()` function of each strategy is invoked once per time step, right after the core simulation
engine completes the day's processing. This ensures that all strategies operate in sync with the simulation's
timeline and react to the market conditions as they evolve.

### Data Structure Assumptions:
One of the core assumptions of Hindsight's design is that the data is organized in a 3-dimensional array of
shape `T x N x J`, where:
- `T` represents the time dimension (e.g., days).
- `N` represents the number of stocks (or other financial instruments).
- `J` represents the number of features or data points available for each stock (e.g., open price, close price,
  volume).

This structure is chosen for its efficiency in vectorized operations, which are crucial for high-performance
backtesting. By structuring the data in this manner, we ensure that each day's data can be processed in parallel,
leveraging NumPy's powerful array operations.

### Usage:
- To create a new strategy, subclass the `Strategy` class and implement the `next()` method, which contains the
  logic for deciding on trades based on the current day's data.
- Additional methods and attributes can be added to the subclass as needed to support more complex strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any

class Strategy(ABC):
    def __init__(self):
        self.params: Dict[str, Any] = {}  # Strategy parameters

    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize strategy-specific variables and parameters.
        This method MUST be implemented by each concrete strategy.
        """
        pass

    @abstractmethod
    def next(self) -> None:
        """
        Core strategy logic. Called for each bar when the strategy is live.
        This method MUST be implemented by each concrete strategy.
        """
        pass

    def buy(self, asset: int, amount: float = None, price: float = None) -> None:
        """
        Place a buy order for a specific asset.
        :param asset: ID of the asset to buy
        :param amount: Amount to buy (optional)
        :param price: Price to buy at (optional, None for market order)
        """
        # Implementation would be provided by the backtesting engine
        pass

    def sell(self, asset: int, amount: float = None, price: float = None) -> None:
        """
        Place a sell order for a specific asset.
        :param asset: ID of the asset to sell
        :param amount: Amount to sell (optional)
        :param price: Price to sell at (optional, None for market order)
        """
        # Implementation would be provided by the backtesting engine
        pass

    def close(self, asset: int) -> None:
        """
        Close the entire position for a specific asset.
        :param asset: ID of the asset to close the position for
        """
        # Implementation would be provided by the backtesting engine
        pass

    @abstractmethod
    def warmup_period(self) -> int:
        """
        Define the warm-up period required before the strategy can trade.
        This method MUST be implemented by each concrete strategy.
        :return: Number of bars required for warm-up
        """
        pass

# Note: Concrete strategy implementations should inherit from this Strategy class
# and implement the required abstract methods (initialize, next, warmup_period).