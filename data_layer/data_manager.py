
import numpy as np
import pandas as pd
from .tensor import Tensor, ReturnsTensor, CharacteristicsTensor 
import jax
import jax.numpy as jnp
from typing import List, Tuple
from functools import partial

from .coords import Coordinates

jax.config.update("jax_enable_x64", True)
import equinox as eqx

import numpy as np
from .tensor import Tensor, ReturnsTensor, CharacteristicsTensor 
import jax
import jax.numpy as jnp
from typing import List, Tuple
from functools import partial

from .coords import Coordinates

jax.config.update("jax_enable_x64", True)
import equinox as eqx

class DataSet(eqx.Module):
    """
    Represents a collection of processed tensors, providing utilities for handling multiple tensors.
    This class is intended to manage data prepared for specific strategies.

    Methods and attributes can be added to facilitate operations such as data splitting,
    feature engineering, and batching.
    """
    def __init__(self):
        pass
        # Initialization code here
        # For example, you might store a list of tensors or a dictionary mapping names to tensors
        
    # Additional methods for loading other types of data

class DataLoader(eqx.Module):
    """
    Responsible for loading data from various sources and creating tensor instances.
    Handles data retrieval, caching, and synchronization across different data sources.
    """
    def __init__(self):
        pass
        # Initialization code here
        # For example, you might set up paths to data sources or initialize caching mechanisms

    @eqx.filter_jit
    def _generate_random_data(self, shape):
        """
        Generates random data for simulation purposes.

        Args:
            shape (Tuple[int, ...]): Shape of the data array to generate.

        Returns:
            jnp.ndarray: Generated random data as a JAX array.
        """
        return jax.random.normal(jax.random.PRNGKey(0), shape, dtype=jnp.float32)

    def simulate_load_returns(self) -> ReturnsTensor:
        """
        Simulates loading returns data from multiple sources.

        Returns:
            ReturnsTensor: A ReturnsTensor containing the simulated data.
        """
        # Simulate data generation
        data = self._generate_random_data((10_000, 3, 1))  # JAX array

        # Define coordinate variables for the tensor
        coord_vars = {
            'time': jnp.arange(10_000),                  # Numerical coordinate: time steps from 0 to 9999
            'asset': ['AAPL', 'GOOG', 'MSFT'],           # Categorical coordinate: asset names
            'feature': ['return']                        # Categorical coordinate: asset features
        }

        # Initialize the Coordinates object with the defined coordinate variables
        coords = Coordinates(coord_vars)

        # Return a ReturnsTensor instance
        return ReturnsTensor(data, coords)
    
    @eqx.filter_jit
    def load_csv(self, filepath) -> jnp.ndarray:
        return jnp.asarray(pd.read_csv(filepath).to_numpy())

    @eqx.filter_jit
    def load_parquet(self, filepath) -> jnp.ndarray:
        return jnp.asarray(pd.read_parquet(filepath).to_numpy())
 