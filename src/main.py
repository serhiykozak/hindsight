from hindsight.core import Environment
from hindsight.broker import Broker
from hindsight.strategy import Strategy
from hindsight.structures import DataFeed
from hindsight.loaders import from_parquet

# Load the data
tensor, permno_to_index, feature_to_index, date_to_index, start_date, end_date, features = from_parquet(
    '../data/raw/dsf.parquet', is_folder=True)

# Make a data feed to simulate price stream
data = DataFeed(data=tensor
                permno_to_index=permno_to_index,
                feature_to_index=feature_to_index,
                date_to_index=date_to_index)

# Spawn a broker
broker = Broker()

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

