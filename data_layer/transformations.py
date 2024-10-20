

@transformation
def normalize_using_median(block: Tensor) -> Tensor:
    """
    Normalizes the time-series in a rolling window using the log of the median of the time-series.
    """
   
    median = block.median('time', slice(-1, 30))
    
    
   raise NotImplementedError("This transformation has not been implemented yet.")