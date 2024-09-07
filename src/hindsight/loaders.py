import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import concurrent.futures

EXCLUSION_LIST = ['date', 'permno', 'permco', 'hsiccd', 'hexcd', 'cusip', 'issuno', 'month']

def _load_parquet_data(folder_path: str) -> pd.DataFrame:
    """
    Load and concatenate parquet files from a folder into a single DataFrame.
    """
    parquet_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.parquet')]
    if not parquet_files:
        raise ValueError(f"No parquet files found in {folder_path}")

    # Load and concatenate all parquet files with parallel processing
    def load_file(file):
        return pd.read_parquet(file)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        dataframes = list(tqdm(executor.map(load_file, parquet_files), total=len(parquet_files), desc="Loading parquet files"))
    
    return pd.concat(dataframes, ignore_index=True)

def _get_valid_permnos(df: pd.DataFrame, min_months: int) -> pd.Index:
    """
    Filter stocks by minimum number of months of data.
    """
    df['month'] = df['date'].dt.to_period('M')
    permno_counts = df.groupby('permno')['month'].nunique()
    return permno_counts[permno_counts >= min_months].index

def _initialize_mappings(unique_permnos, features, date_range) -> tuple:
    """
    Initialize mappings for permnos, features, and dates.
    """
    permno_to_index = {permno: idx for idx, permno in enumerate(unique_permnos)}
    feature_to_index = {feature: idx for idx, feature in enumerate(features)}
    feature_to_index['return'] = len(features)  # Add the return feature
    date_to_index = {date: idx for idx, date in enumerate(date_range)}

    return permno_to_index, feature_to_index, date_to_index

def _populate_tensor(df, features, tensor_shape, permno_to_index, date_to_index):
    """
    Populate the 3D tensor with feature and return data.
    """
    T, N, J = tensor_shape
    tensor = np.full((T, N, J), np.nan, dtype=np.float64)

    df['date_idx'] = df['date'].map(date_to_index).values
    df['permno_idx'] = df['permno'].map(permno_to_index).values

    # Combine features and return values
    feature_values = df[features].values
    return_values = df['ret'].values.reshape(-1, 1)
    feature_values = np.hstack((feature_values, return_values))
    
    # Populate tensor using advanced indexing
    tensor[df['date_idx'], df['permno_idx']] = feature_values

    tensor[np.isinf(tensor)] = np.nan  # Handle infinite values

    return tensor

def _process_dataframe(df: pd.DataFrame, min_months: int) -> tuple:
    """
    Process the dataframe to filter valid stocks and extract unique permnos and features.
    """
    df['date'] = pd.to_datetime(df['date'], errors='coerce')  # Ensure date is datetime

    # Filter valid permnos
    valid_permnos = _get_valid_permnos(df, min_months)
    df = df[df['permno'].isin(valid_permnos)]

    # Identify unique permnos and features
    unique_permnos = sorted(valid_permnos)
    features = sorted([col for col in df.columns if col not in EXCLUSION_LIST])

    # Define the date range
    start_date = df['date'].min()
    end_date = df['date'].max()
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    return df, unique_permnos, features, date_range, start_date, end_date

def _create_tensor(df: pd.DataFrame, unique_permnos, features, date_range):
    """
    Create the 3D tensor and related mappings for the processed dataframe.
    """
    tensor_shape = (len(date_range), len(unique_permnos), len(features) + 1)  # +1 for the return feature

    print('Now mapping...')

    # Initialize mappings
    permno_to_index, feature_to_index, date_to_index = _initialize_mappings(unique_permnos, features, date_range)

    print('Now populating tensor...')

    # Populate the tensor
    tensor = _populate_tensor(df, features, tensor_shape, permno_to_index, date_to_index)

    return tensor, permno_to_index, feature_to_index, date_to_index, features + ['return']

def from_parquet(path: str, min_months: int = 60, is_folder: bool = False):
    """
    Load data from a single parquet file or multiple parquet files from a folder,
    filter stocks with insufficient data, and transform it into a numpy tensor.
    """
    if is_folder:
        df = _load_parquet_data(path)
    else:
        df = pd.read_parquet(path)

    df, unique_permnos, features, date_range, start_date, end_date = _process_dataframe(df, min_months)

    tensor, permno_to_index, feature_to_index, date_to_index, feature_list = _create_tensor(df, unique_permnos, features, date_range)

    return tensor, permno_to_index, feature_to_index, date_to_index, start_date, end_date, feature_list
