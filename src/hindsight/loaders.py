import pandas as pd
import numpy as np
import os
import concurrent.futures
from tqdm import tqdm
import time

EXCLUSION_LIST = ['date', 'permno', 'permco', 'hsiccd', 'hexcd', 'cusip', 'issuno']

def _load_parquet_data(folder_path: str) -> pd.DataFrame:
    """
    Load and concatenate parquet files from a folder into a single DataFrame.
    """
    start_time = time.time()
    parquet_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith('.parquet')
    ]
    if not parquet_files:
        raise ValueError(f"No parquet files found in {folder_path}")

    def load_file(file):
        return pd.read_parquet(file)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        dataframes = list(
            tqdm(
                executor.map(load_file, parquet_files),
                total=len(parquet_files),
                desc="Loading parquet files"
            )
        )

    # Remove duplicate entries based on 'date' and 'permno'
    df = pd.concat(dataframes, ignore_index=True)
    df = df.drop_duplicates(subset=['date', 'permno'], keep='last').reset_index(drop=True)

    end_time = time.time()
    print(f"Data loading completed in {end_time - start_time:.2f} seconds.")
    return df

def _process_dataframe(df: pd.DataFrame) -> tuple:
    """
    Process the dataframe to extract unique permnos, features, and dates.
    """
    start_time = time.time()
    # Ensure 'date' is datetime and normalized
    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.normalize()
    df = df.dropna(subset=['date', 'permno'])  # Drop rows with invalid dates or permnos

    # Identify unique permnos and features
    unique_permnos = df['permno'].unique()
    unique_permnos.sort()
    features = [col for col in df.columns if col not in EXCLUSION_LIST]

    # Keep only necessary columns to reduce memory usage
    df = df[['date', 'permno'] + features]

    # Define continuous date range
    start_date = df['date'].min()
    end_date = df['date'].max()
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    end_time = time.time()
    print(f"Data processing completed in {end_time - start_time:.2f} seconds.")
    return df, unique_permnos, features, date_range, start_date, end_date

def _initialize_mappings(unique_permnos, features, date_range) -> tuple:
    """
    Initialize mappings for permnos, features, and dates.
    """
    start_time = time.time()
    # Create mappings from permnos and dates to indices
    permno_to_index = {permno: idx for idx, permno in enumerate(unique_permnos)}
    feature_to_index = {feature: idx for idx, feature in enumerate(features)}
    date_to_index = {date: idx for idx, date in enumerate(date_range)}

    end_time = time.time()
    print(f"Mappings initialization completed in {end_time - start_time:.2f} seconds.")
    return permno_to_index, feature_to_index, date_to_index

def _populate_tensor(df, features, tensor_shape, unique_permnos, date_range):
    """
    Populate the 3D tensor with feature data using vectorized operations.
    """
    start_time = time.time()
    T, N, J = tensor_shape

    # Map dates and permnos using Index objects
    date_index = pd.Index(date_range)
    permno_index = pd.Index(unique_permnos)

    # Get indices using get_indexer
    df['date_idx'] = date_index.get_indexer(df['date'])
    df['permno_idx'] = permno_index.get_indexer(df['permno'])

    # Keep only valid indices (>=0)
    valid_mask = (df['date_idx'] >= 0) & (df['permno_idx'] >= 0)
    df_valid = df.loc[valid_mask, :]

    # Extract indices and feature values as NumPy arrays
    date_indices = df_valid['date_idx'].values
    permno_indices = df_valid['permno_idx'].values
    feature_values = df_valid[features].values.astype(np.float32)

    # Initialize tensor
    tensor = np.full((T, N, J), np.nan, dtype=np.float32)

    # Use advanced indexing to assign values
    tensor[date_indices, permno_indices, :] = feature_values

    # Handle infinite values
    tensor[np.isinf(tensor)] = np.nan

    end_time = time.time()
    print(f"Tensor population completed in {end_time - start_time:.2f} seconds.")
    return tensor

def _create_tensor(df: pd.DataFrame, unique_permnos, features, date_range):
    """
    Create the 3D tensor and related mappings for the processed dataframe.
    """
    print('Now initializing mappings...')
    permno_to_index, feature_to_index, date_to_index = _initialize_mappings(
        unique_permnos, features, date_range
    )

    print('Now populating tensor...')
    tensor_shape = (len(date_range), len(unique_permnos), len(features))
    tensor = _populate_tensor(
        df, features, tensor_shape, unique_permnos, date_range
    )

    return tensor, permno_to_index, feature_to_index, date_to_index, features

def from_parquet(path: str, is_folder: bool = False):
    """
    Load data from a single parquet file or multiple parquet files from a folder,
    and transform it into a numpy tensor.
    """
    total_start_time = time.time()

    if is_folder:
        df = _load_parquet_data(path)
    else:
        start_time = time.time()
        df = pd.read_parquet(path)
        end_time = time.time()
        print(f"Data loading completed in {end_time - start_time:.2f} seconds.")

    df, unique_permnos, features, date_range, start_date, end_date = _process_dataframe(df)

    tensor, permno_to_index, feature_to_index, date_to_index, feature_list = _create_tensor(
        df, unique_permnos, features, date_range
    )

    total_end_time = time.time()
    print(f"Total time taken: {total_end_time - total_start_time:.2f} seconds.")
    return tensor, permno_to_index, feature_to_index, date_to_index, start_date, end_date, feature_list