from hindsight.core import Environment
from hindsight.broker import Broker
from hindsight.strategy import Strategy
from hindsight.structures import DataFeed

import numpy as np

# This is an evolving visualization

# load bunch of data and form dataset
tensor = np.array([[[]]])
data = DataFeed(data=tensor)

# Spawn a broker & a strategy
broker = Broker(some stuff...)

class SomeStrategy(Strategy):

    def warmup_period(self) -> int:
        pass

    def next(self) -> None:
        pass

    def initialize(self) -> None:
        pass

# Compile the strategy
strategy = SomeStrategy()

system = System([strategy, ...])

# Create an environment
env = Environment(broker = broker, data = data, system = system)

# run the backtest
env.run()