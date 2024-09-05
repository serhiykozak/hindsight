import numpy as np
from typing import List

from broker import Broker
from structures import DataFeed

class Environment:

    def __init__(self,
                 broker: Broker,
                 data: DataFeed,):

        self.broker = broker
        self.datas = data

