# data_layer/coords.py

import numpy as np
import equinox as eqx
from typing import Mapping, Any, Iterator
import jax.numpy as jnp

class Coordinates(eqx.Module):
    """
    Manages coordinate variables associated with tensor dimensions.
    Supports both numerical and non-numerical data types.

    Attributes:
        variables (Mapping[str, Any]): 
            A mapping from dimension names to their coordinate data.
            - Numerical data types are converted to numpy arrays for compatibility with JAX transformations.
            - Non-numerical data types (e.g., strings) are stored as-is to maintain flexibility.
    """
    variables: Mapping[str, Any] = eqx.field(static=True)
    
    def __post_init__(self):
        """
        Processes the coordinate variables after initialization:
        - Converts numerical data types to numpy arrays for optimized computations.
        - Keeps non-numerical data types unchanged to preserve information such as categorical labels.
        """
        processed_vars = {}
        for name, var in self.variables.items():
            if isinstance(var, jnp.ndarray):
                # Convert JAX arrays to numpy arrays
                processed_vars[name] = np.array(var)
            elif isinstance(var, np.ndarray):
                # Keep numpy arrays as they are
                processed_vars[name] = var
            elif self._is_numerical(var):
                # Convert numerical lists or tuples to numpy arrays
                processed_vars[name] = np.array(var)
            else:
                # Keep non-numerical data types unchanged
                processed_vars[name] = var
        # Update the 'variables' attribute with the processed variables
        object.__setattr__(self, 'variables', processed_vars)
    
    @staticmethod
    def _is_numerical(var: Any) -> bool:
        """
        Determines if the provided variable is numerical.

        Args:
            var (Any): The variable to check.

        Returns:
            bool: True if the variable is numerical, False otherwise.
        """
        if isinstance(var, (list, tuple)):
            # Check if all elements in the list or tuple are numerical types
            return all(isinstance(v, (int, float, complex)) for v in var)
        # Check if the variable is a numerical numpy array or a single numerical value
        return isinstance(var, (int, float, complex, np.ndarray))
    
    def __repr__(self) -> str:
        """
        Provides a string representation of the Coordinates object.

        Returns:
            str: String representation showing all coordinate variables.
        """
        return f"Coordinates(variables={self.variables!r})"
    
    def __getitem__(self, key: str) -> Any:
        """
        Allows retrieval of coordinate variables by their dimension name.

        Args:
            key (str): The name of the dimension.

        Returns:
            Any: The coordinate data associated with the dimension.
        """
        return self.variables[key]
    
    def __iter__(self) -> Iterator[str]:
        """
        Returns an iterator over the dimension names.

        Returns:
            Iterator[str]: An iterator for the dimension names.
        """
        return iter(self.variables)
    
    def __len__(self) -> int:
        """
        Returns the number of coordinate variables.

        Returns:
            int: The count of coordinate variables.
        """
        return len(self.variables)
    
    def __hash__(self) -> int:
        """
        Computes a hash based on the coordinate variable names and their data.
        This allows Coordinates instances to be used in hashed collections like sets or as dictionary keys.

        Returns:
            int: The computed hash value.
        """
        hashable_items = []
        for name, var in self.variables.items():
            if isinstance(var, np.ndarray):
                # Convert numpy arrays to bytes for hashing
                hashable_items.append((name, var.tobytes()))
            else:
                # Convert lists to tuples to make them hashable
                hashable_items.append((name, tuple(var)))
        # Use frozenset to ensure the hash is independent of the order of variables
        return hash(frozenset(hashable_items))
    
    def __eq__(self, other: Any) -> bool:
        """
        Checks equality with another Coordinates instance.

        Args:
            other (Any): The object to compare with.

        Returns:
            bool: True if equal, False otherwise.
        """
        if self is other:
            return True
        if not isinstance(other, Coordinates):
            return NotImplemented
        if self.variables.keys() != other.variables.keys():
            return False
        # Compare each coordinate variable for equality
        for name in self.variables:
            var1 = self.variables[name]
            var2 = other.variables[name]
            if isinstance(var1, np.ndarray) and isinstance(var2, np.ndarray):
                # Use numpy's array_equal for numerical arrays
                if not np.array_equal(var1, var2):
                    return False
            else:
                # Direct comparison for non-numerical data types
                if var1 != var2:
                    return False
        return True
