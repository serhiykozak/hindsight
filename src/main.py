# from hindsight.core import Environment
# from hindsight.broker import Broker
# from hindsight.strategy import Strategy
# from hindsight.structures import DataFeed
#
# import numpy as np
#
# # This is an evolving visualization
#
# # load bunch of data and form dataset
# tensor = np.array([[[]]])
# data = DataFeed(data=tensor)
#
# # Spawn a broker & a strategy
# broker = Broker(some stuff...)
#
# class SomeStrategy(Strategy):
#
#     def warmup_period(self) -> int:
#         pass
#
#     def next(self) -> None:
#         pass
#
#     def initialize(self) -> None:
#         pass
#
# # Compile the strategy
# strategy = SomeStrategy()
#
# system = System([strategy, ...])
#
# # Create an environment
# env = Environment(broker = broker, data = data, system = system)
#
# # run the backtest
# env.run()
import random

from hindsight.loaders import from_parquet
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use('dark_background')

# Load the data
tensor, permno_to_index, feature_to_index, date_to_index, start_date, end_date, features = from_parquet('../data/raw/dsf.parquet',
                                                                                                         is_folder=True)

# Select a random permno
# random_permno = random.choice(list(permno_to_index.keys()))
random_permno = 14593
permno_index = permno_to_index[random_permno]

# Create a date range upfront
dates = pd.date_range(start=start_date, end=end_date, freq='D')

# Fetch the closing price data efficiently
closing_prices = tensor[:, permno_index, feature_to_index['prc']]

# Plot the data
plt.figure(figsize=(10, 6))  # Optimized figure size for better visibility
plt.plot(dates, closing_prices, linewidth=1)
plt.xlabel('Date')
plt.ylabel('Closing price (USD)')
plt.title(f'Closing Price of Permno {random_permno}')
plt.tight_layout()  # Optimized layout

# Save the figure efficiently
plt.savefig('apple_close_prices_(crsp).png', dpi=300, bbox_inches='tight')
plt.close()



















