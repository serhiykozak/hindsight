# hindsight/main.py

# Importing necessary classes from the data_layer and coords modules
from data_layer.tensor import Tensor, ReturnsTensor, CharacteristicsTensor
from data_layer.coords import Coordinates
from data_layer.data_manager import DataManager
# Importing JAX's NumPy module for numerical operations
import jax.numpy as jnp

def normalize_returns(block: Tensor) -> Tensor:
    """
    Normalizes the latest returns in a rolling window using the mean and standard deviation.
    """
    # Compute the mean and standard deviation over the 'time' dimension
    mean = block.mean('time')       # Shape: (1, asset, feature)
    std = block.std('time')         # Shape: (1, asset, feature)

    # Select the latest data point in the time window
    latest_data = block.select('time', -1)  # Shape: (1, asset, feature)

    # Normalize the latest data point
    normalized_data = (latest_data - mean) / std  # Shape: (1, asset, feature)

    return normalized_data  # Returns a ReturnsTensor instance

def main():
    """
    Main function to demonstrate the initialization and usage of Coordinates.
    This function sets up coordinate variables, initializes a Coordinates instance,
    and prints out the coordinate details.
    """
    
    dm = DataManager()
    returns = dm.simulate_load_returns()
    
    # Define the window size for rolling normalization
    window_size = 252

    # Apply the normalize_returns function over rolling windows using u_roll

    # Print the tensor
    print(returns)      
    
    
if __name__ == "__main__":
    main()

# 
# Returns Tensor -> Coordinates=Coordinates(time=[u32int..... 1D
# Characteristics Tensor -> Coordinates=Coordinates(time=[u32int....., asset=[u32int....., feature=[str.....]) 10A
# Characteristics Tensor -> Coordinates=Coordinates(time=[u32int....., asset=[u32int....., feature=[str.....]) 15Q
# 
# 
# 
# 
# 
# 
# 
# 
# 
    # 
Tensor (1, N, 10 + 1 + 15)