import pandas as pd
import numpy as np
from tqdm import tqdm
import os

# Exclusion list for features
EXCLUSION_LIST = ['date', 'permno', 'permco', 'hsiccd', 'hexcd', 'cusip', 'issuno', 'month']


def _load_parquet_data(folder_path: str) -> pd.DataFrame:
    """
    Load and concatenate parquet files from a folder into a single DataFrame.

    Args:
    folder_path (str): Path to the folder containing parquet files.

    Returns:
    pd.DataFrame: Concatenated DataFrame with all the parquet data.
    """
    parquet_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.parquet')]

    if not parquet_files:
        raise ValueError(f"No parquet files found in {folder_path}")

    # Load and concatenate all parquet files with a progress bar
    dataframes = []
    for f in tqdm(parquet_files, desc="Loading parquet files"):
        dataframes.append(pd.read_parquet(f))

    return pd.concat(dataframes, ignore_index=True)


def _get_valid_permnos(df: pd.DataFrame, min_months: int) -> pd.Index:
    """
    Filter stocks by minimum number of months of data.

    Args:
    df (pd.DataFrame): DataFrame containing the stock data.
    min_months (int): Minimum number of months required for a stock.

    Returns:
    pd.Index: Index of valid permnos with sufficient data.
    """
    df['month'] = df['date'].dt.to_period('M')
    permno_counts = df.groupby('permno')['month'].nunique()
    return permno_counts[permno_counts >= min_months].index


def _initialize_mappings(unique_permnos, features, date_range) -> tuple:
    """
    Initialize mappings for permnos, features, and dates.

    Args:
    unique_permnos (list): List of unique permnos.
    features (list): List of features.
    date_range (pd.DatetimeIndex): Range of dates.

    Returns:
    tuple: Dictionaries for permno_to_index, feature_to_index, and date_to_index.
    """
    permno_to_index = {permno: idx for idx, permno in enumerate(unique_permnos)}
    feature_to_index = {feature: idx for idx, feature in enumerate(features)}
    feature_to_index['return'] = len(features)  # Add the return feature
    date_to_index = {date: idx for idx, date in enumerate(date_range)}

    return permno_to_index, feature_to_index, date_to_index


def _populate_tensor(df, features, tensor_shape, permno_to_index, date_to_index):
    """
    Populate the 3D tensor with feature and return data.

    Args:
    df (pd.DataFrame): DataFrame containing the stock data.
    features (list): List of feature columns.
    tensor_shape (tuple): Shape of the tensor (T, N, J).
    permno_to_index (dict): Mapping of permnos to tensor indices.
    date_to_index (dict): Mapping of dates to tensor indices.

    Returns:
    np.ndarray: The populated tensor.
    """
    T, N, J = tensor_shape
    tensor = np.full((T, N, J), np.nan, dtype=np.float64)

    df['date_idx'] = df['date'].map(date_to_index)
    df['permno_idx'] = df['permno'].map(permno_to_index)

    # Combine features and return values
    feature_values = df[features].values
    return_values = df['ret'].values.reshape(-1, 1)
    feature_values = np.hstack((feature_values, return_values))

    # Populate tensor using numpy advanced indexing with a progress bar
    for idx in tqdm(range(df.shape[0]), desc="Populating tensor"):
        date_idx = df.iloc[idx]['date_idx']
        permno_idx = df.iloc[idx]['permno_idx']
        tensor[date_idx, permno_idx] = feature_values[idx]

    # Handle infinite values by replacing them with NaN
    tensor[np.isinf(tensor)] = np.nan

    return tensor


def _process_dataframe(df: pd.DataFrame, min_months: int) -> tuple:
    """
    Process the dataframe to filter valid stocks and extract unique permnos and features.

    Args:
    df (pd.DataFrame): DataFrame containing stock data.
    min_months (int): Minimum number of months required for a stock.

    Returns:
    tuple: Processed dataframe, unique permnos, features, and date range.
    """
    df['date'] = pd.to_datetime(df['date'])  # Ensure date is datetime

    # Get the start and end dates
    start_date = df['date'].min()
    end_date = df['date'].max()

    # Filter valid permnos
    valid_permnos = _get_valid_permnos(df, min_months)
    df = df[df['permno'].isin(valid_permnos)]

    # Identify unique permnos and features
    unique_permnos = sorted(valid_permnos)
    features = sorted([col for col in df.columns if col not in EXCLUSION_LIST])

    # Define the date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    return df, unique_permnos, features, date_range, start_date, end_date


def _create_tensor(df: pd.DataFrame, unique_permnos, features, date_range):
    """
    Create the 3D tensor and related mappings for the processed dataframe.

    Args:
    df (pd.DataFrame): Processed stock data.
    unique_permnos (list): List of unique permnos.
    features (list): List of features.
    date_range (pd.DatetimeIndex): Range of dates.

    Returns:
    tuple: The populated tensor, mappings, and metadata.
    """
    # Define tensor shape
    tensor_shape = (len(date_range), len(unique_permnos), len(features) + 1)  # +1 for the return feature

    # Initialize mappings
    permno_to_index, feature_to_index, date_to_index = _initialize_mappings(unique_permnos, features, date_range)

    # Populate the tensor
    tensor = _populate_tensor(df, features, tensor_shape, permno_to_index, date_to_index)

    return tensor, permno_to_index, feature_to_index, date_to_index, features + ['return']


def from_parquet(path: str, min_months: int = 60, is_folder: bool = False):
    """
    Load data from a single parquet file or multiple parquet files from a folder,
    filter stocks with insufficient data, and transform it into a numpy tensor.

    Args:
    path_or_folder (str): Path to the parquet file or folder containing parquet files.
    min_months (int, optional): The minimum number of months required for a stock. Defaults to 60.
    is_folder (bool, optional): Whether the input is a folder containing parquet files. Defaults to False.

    Returns:
    dict: A dictionary containing the tensor, mappings, date range, and features.
    """
    if is_folder:
        df = _load_parquet_data(path)
    else:
        df = pd.read_parquet(path)

    # Process the dataframe to get valid permnos, features, and date range
    df, unique_permnos, features, date_range, start_date, end_date = _process_dataframe(df, min_months)

    # Create tensor and get the necessary mappings
    tensor, permno_to_index, feature_to_index, date_to_index, feature_list = _create_tensor(df, unique_permnos, features, date_range)

    return tensor, permno_to_index, feature_to_index, date_to_index, start_date, end_date, feature_list
