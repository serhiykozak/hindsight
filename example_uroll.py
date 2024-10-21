# hindsight/main.py

# Import necessary modules and classes
from data_layer.tensor import CharacteristicsTensor
from data_layer.coords import Coordinates
import jax.numpy as jnp
import jax
import numpy as np
import pandas as pd
import timeit
from functools import partial

# This example demonstrates the use of JAX for efficient numerical computations,
# particularly in the context of financial data processing.

# JAX is a library for high-performance numerical computing, especially suited
# for machine learning tasks. It provides automatic differentiation and can
# leverage GPU/TPU acceleration.

def generate_sine_wave(time_steps, frequency, amplitude=1.0):
    """
    Generate a sine wave using JAX's numpy (jnp).
    
    JAX's numpy is similar to regular numpy but allows for JIT compilation
    and automatic differentiation. This function creates a simple sine wave
    which will be used to simulate financial data.
    
    Args:
    time_steps (int): Number of time steps (data points) in the wave
    frequency (float): Frequency of the sine wave
    amplitude (float): Amplitude of the sine wave (default 1.0)
    
    Returns:
    jnp.array: JAX numpy array containing the sine wave data
    """
    t = jnp.arange(time_steps)
    return amplitude * jnp.sin(2 * jnp.pi * frequency * t / time_steps)

def main():
    # Set parameters for data generation
    num_time_steps = 1000
    num_assets = 5_000
    num_features = 5
    
    # Generate synthetic data using sine waves
    # This simulates multi-dimensional financial data (time, assets, features)
    data = jnp.zeros((num_time_steps, num_assets, num_features))
    for asset in range(num_assets):
        for feature in range(num_features):
            # Use JAX's random number generation for reproducibility
            frequency = jax.random.uniform(jax.random.PRNGKey(asset * num_features + feature), 
                                           minval=0.1, maxval=10.0)
            # Update the data array using JAX's functional update syntax
            data = data.at[:, asset, feature].set(generate_sine_wave(num_time_steps, frequency))
    
    # Define feature names for our simulated financial data
    feature_names = ('open', 'high', 'low', 'close', 'volume')
    
    # Create coordinate variables for each dimension of our data
    # This helps in labeling and accessing specific parts of the data
    coord_vars = {
        'time': np.arange(num_time_steps),
        'asset': np.arange(num_assets),
        'feature': feature_names
    }
    
    # Create a Coordinates object to manage our coordinate variables
    # This object is defined in the data_layer/coords.py file
    coords = Coordinates(coord_vars)

    # Create a CharacteristicsTensor instance
    # This is a custom tensor class defined in data_layer/tensor.py
    # It's designed to handle multi-dimensional financial data efficiently
    characteristics_tensor = CharacteristicsTensor(
        data=data,
        dimensions=('time', 'asset', 'feature'),
        feature_names=feature_names,
        Coordinates=coords
    )

    # Demonstrate slicing operations on the tensor
    # Select time steps 0 to 9 (10 steps in total)
    selected_tensor = characteristics_tensor.select('time', slice(0, 10))
    print("\n\nShape of selected tensor:", selected_tensor.data.shape)
    print("Time coordinates of selected tensor:", selected_tensor.Coordinates.variables['time'])  
    print("Feature names:", selected_tensor.feature_names) 
    
    # Select only the 'close' feature
    close_tensor = selected_tensor.select('feature', 'close')
    print("\nShape of close tensor:", close_tensor.data.shape)
    print("Feature names of close tensor:", close_tensor.feature_names)   
    
    # Demonstrate aggregation operations
    # Sum the 'close' feature over the time dimension
    close_sum_tensor = selected_tensor.select('feature', 'close').sum('time')
    print("\nShape of close sum tensor:", close_sum_tensor.data.shape)
    print("Feature names of close sum tensor:", close_sum_tensor.feature_names)   
    print("Close sum tensor data:", close_sum_tensor.data)
    
    # Select the 'close' feature from the original tensor for further processing
    close_tensor = characteristics_tensor.select('feature', 'close')
    
    # Define a function to compute Exponential Moving Average (EMA)
    # This function will be used with the u_roll method for efficient computation
    @partial(jax.jit, static_argnames=['window_size'])
    def ema(i: int, state, window_size: int):
        """
        Compute the Exponential Moving Average (EMA) for a given window.
        
        This function is designed to work with JAX's JIT compilation and
        the u_roll method defined in the Tensor class. It computes the EMA
        efficiently over a rolling window of data.
        
        Args:
        i (int): Current index in the time series
        state (tuple): Contains current values, carry (previous EMA), and data block
        window_size (int): Size of the moving window
        
        Returns:
        tuple: Updated state (new EMA value, carry, and data block)
        """
        curr_values, carry, block = state
        
        # Initialize the first value
        if (carry == None):
            # Compute the sum of the first window
            current_window_sum = block[:window_size].reshape(-1, 
                                                             block.shape[1], 
                                                             block.shape[2]).sum(axis=0)
            
            # Set the first EMA value as the average of the first window
            values = curr_values.at[0].set(current_window_sum * (1/window_size))
            
            return (values, current_window_sum * (1/window_size), block)
        
        # Get the current price
        current_price = block[i]
        
        # Compute the new EMA
        # EMA = α * current_price + (1 - α) * previous_EMA
        # where α = 1 / (window_size)
        new_ema = (1 / (window_size)) * current_price + (1 - (1 / (window_size))) * carry
        
        return (new_ema, carry, block)
        
    # Benchmark comparison between custom u_roll method and pandas
    # This demonstrates the performance benefits of our custom implementation
    
    def benchmark_u_roll():
        return close_tensor.u_roll(10, ema)
    
    def benchmark_pandas():
        df = pd.DataFrame(close_tensor.data.squeeze())
        return df.ewm(span=10, adjust=False).mean().values
    
    # Run benchmarks
    u_roll_time = timeit.timeit(benchmark_u_roll, number=10)
    pandas_time = timeit.timeit(benchmark_pandas, number=10)
    
    print("\nBenchmark results (average of 10 runs):")
    print(f"u_roll method: {u_roll_time:.4f} seconds")
    print(f"pandas method: {pandas_time:.4f} seconds")
    
    # Compute EMA using u_roll for further analysis
    compute_ema = close_tensor.u_roll(10, ema)
    
    print("\nShape of computed EMA tensor:", compute_ema.data.shape)
    print("Feature names of computed EMA tensor:", compute_ema.feature_names)
    
    # Verify results by comparing with pandas implementation
    pandas_ema = benchmark_pandas()
    u_roll_ema = compute_ema.data.squeeze()
    
    print("\nVerification:")
    print("Max absolute difference:", jnp.max(jnp.abs(pandas_ema - u_roll_ema)))
    print("Are results close?", jnp.allclose(pandas_ema, u_roll_ema, atol=1e-5))
    print("First 10 values of pandas_ema:", pandas_ema[:10])
    print("First 10 values of u_roll_ema:", u_roll_ema[:10])
    print("First person to debug why this is happening wins a prize!")
    
if __name__ == "__main__":
    main()
