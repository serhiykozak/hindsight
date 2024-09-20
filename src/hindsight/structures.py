"""
structures.py

This file defines various structure classes required for the core operations. These structures include
(but are not limited to) historical data feeds (DataFeed) and position classes. They form the fundamental
objects involved in the environment.

The primary class in this file is DataFeed, which handles three-dimensional data typically used
in financial or time-series analysis.
"""

import numpy as np
from typing import List, Tuple, Dict, Any

from src.hindsight.strategy import Strategy


class DataFeed:
    """
    A class to handle and provide access to three-dimensional data, typically used for time-series financial data.

    The data is expected to be a numpy array with shape (T, N, J), where:
    T represents the time dimension (e.g., days, hours)
    N represents the assets or entities
    J represents the features or characteristics of each asset

    Example:
        If data[50, 5, 0] represents the closing price, then this would be:
        The closing price (0th feature) of the 5th asset on the 50th day.

    This can be conceptualized as a 3-dimensional tensor, where each "row" (first index)
    represents all assets and their features for a specific point in time.
    """

    def __init__(self, data: np.ndarray, feature_to_index, permno_to_index, date_to_index):
        """
        Initialize the DataFeed with the provided data.

        Args:
            data (np.ndarray): A 3-dimensional numpy array of shape (T, N, J)

        Raises:
            TypeError: If the provided data is not a numpy array
            ValueError: If the provided data is not 3-dimensional
        """
        # Validate that the input is a numpy array
        if not isinstance(data, np.ndarray):
            raise TypeError("Data must be an instance of np.ndarray.")

        # Validate that the input is 3-dimensional
        if data.ndim != 3:
            raise ValueError("Data must be of shape T X N X J (3-dimensional).")

        # Store the data and its dimensions
        self.data = data
        self.T, self.N, self.J = data.shape
        
        self.feature_to_index = feature_to_index 
        self.permno_to_index = permno_to_index
        self.date_to_index = date_to_index

    def get(self, t: int = 0, n: int = None, j: int = None) -> np.ndarray:
        """
        Retrieve data from the DataFeed.

        This method provides flexible data access:
        - If only t is provided, it returns all assets and features for that time.
        - If t and n are provided, it returns all features for the specified asset at that time.
        - If all three indices are provided, it returns a specific value.

        Args:
            t (int): The time index. Defaults to 0.
            n (int, optional): The asset index. If None, all assets are returned.
            j (int, optional): The feature index. If None, all features are returned.

        Returns:
            np.ndarray: The requested data slice.

        Note:
            This method leverages NumPy's advanced indexing for efficient data retrieval.
        """
        return self.data[t, n, j]
