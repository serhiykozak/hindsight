
import numpy as np
from .tensor import Tensor, ReturnsTensor, CharacteristicsTensor 
import jax
import jax.numpy as jnp
from typing import List, Tuple
from functools import partial

from .coords import Coordinates

jax.config.update("jax_enable_x64", True)
import equinox as eqx
    
class DataManager(eqx.Module):
    def __init__(self):
        pass

    @eqx.filter_jit
    def _generate_random_data(self, shape):
        return jax.random.normal(jax.random.PRNGKey(0), shape, dtype=jnp.float32)

    def simulate_load_returns(self) -> ReturnsTensor:
        """
        Simulates loading returns data from multiple sources.
        :return: ReturnsTensor containing the data.
        """

        data = self._generate_random_data((10_000, 3, 1))  # JAX array
        # Define coordinate variables for the tensor
        coord_vars = {
            'time': jnp.arange(10_000),                  # Numerical coordinate: time steps from 0 to 9999
            'asset': ['AAPL', 'GOOG', 'MSFT'],           # Categorical coordinate: asset names
            'feature': ['return']                        # Categorical coordinate: asset features
        }

        # Initialize the Coordinates object with the defined coordinate variables
        coords = Coordinates(coord_vars)

        return ReturnsTensor(data, coords)
