"""
============================================================================
Helper Utilities Module
Contains common utility functions for data manipulation, validation,
and general-purpose operations
============================================================================
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional, Union
from pathlib import Path
import time
from functools import wraps

from .logger import get_logger

logger = get_logger()


def set_random_seeds(seed: int = 42):
    """
    Set random seeds for reproducibility
    
    Args:
        seed: Random seed value
    """
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    
    logger.info(f"Random seeds set to: {seed}")


def timer(func):
    """
    Decorator to measure function execution time
    
    Usage:
        @timer
        def my_function():
            ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info(f"{func.__name__} completed in {duration:.2f}s")
        return result
    
    return wrapper


def validate_dataframe(df: pd.DataFrame, 
                       required_columns: Optional[List[str]] = None,
                       min_rows: int = 1) -> Tuple[bool, str]:
    """
    Validate DataFrame structure and content
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        min_rows: Minimum number of rows required
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check if DataFrame is empty
    if df is None or len(df) == 0:
        return False, "DataFrame is empty"
    
    # Check minimum rows
    if len(df) < min_rows:
        return False, f"DataFrame has {len(df)} rows, minimum {min_rows} required"
    
    # Check required columns
    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            return False, f"Missing columns: {missing_cols}"
    
    return True, "Valid"


def get_missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get summary of missing values in DataFrame
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with missing value statistics
    """
    missing_count = df.isnull().sum()
    missing_percent = (missing_count / len(df)) * 100
    
    summary = pd.DataFrame({
        'Missing_Count': missing_count,
        'Missing_Percent': missing_percent
    })
    
    summary = summary[summary['Missing_Count'] > 0].sort_values(
        'Missing_Count', ascending=False
    )
    
    return summary


def detect_outliers_iqr(data: np.ndarray, 
                        threshold: float = 1.5) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Detect outliers using IQR method
    
    Args:
        data: Input array
        threshold: IQR multiplier (default: 1.5)
    
    Returns:
        Tuple of (outlier_mask, statistics)
    """
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    outlier_mask = (data < lower_bound) | (data > upper_bound)
    
    stats = {
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'n_outliers': np.sum(outlier_mask),
        'outlier_percent': (np.sum(outlier_mask) / len(data)) * 100
    }
    
    return outlier_mask, stats


def detect_outliers_zscore(data: np.ndarray, 
                           threshold: float = 3.0) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Detect outliers using Z-score method
    
    Args:
        data: Input array
        threshold: Z-score threshold (default: 3.0)
    
    Returns:
        Tuple of (outlier_mask, statistics)
    """
    mean = np.mean(data)
    std = np.std(data)
    
    z_scores = np.abs((data - mean) / std)
    outlier_mask = z_scores > threshold
    
    stats = {
        'mean': mean,
        'std': std,
        'threshold': threshold,
        'n_outliers': np.sum(outlier_mask),
        'outlier_percent': (np.sum(outlier_mask) / len(data)) * 100
    }
    
    return outlier_mask, stats


def create_sequences(data: np.ndarray, 
                     lookback: int, 
                     forecast_horizon: int = 1,
                     stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for time-series forecasting
    
    Args:
        data: Input time-series data (shape: [n_timesteps, n_features])
        lookback: Number of previous timesteps to use
        forecast_horizon: Number of future timesteps to predict
        stride: Step size for sliding window
    
    Returns:
        Tuple of (X, y) where X is input sequences and y is targets
    """
    X, y = [], []
    
    for i in range(0, len(data) - lookback - forecast_horizon + 1, stride):
        X.append(data[i:i + lookback])
        y.append(data[i + lookback:i + lookback + forecast_horizon])
    
    return np.array(X), np.array(y)


def normalize_data(data: np.ndarray, 
                   method: str = 'minmax',
                   feature_range: Tuple[float, float] = (0, 1)) -> Tuple[np.ndarray, Any]:
    """
    Normalize data using specified method
    
    Args:
        data: Input data
        method: 'minmax' or 'standard'
        feature_range: Range for minmax scaling
    
    Returns:
        Tuple of (normalized_data, scaler)
    """
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    
    if method == 'minmax':
        scaler = MinMaxScaler(feature_range=feature_range)
    elif method == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    # Handle 1D arrays
    if data.ndim == 1:
        data = data.reshape(-1, 1)
        normalized = scaler.fit_transform(data).flatten()
    else:
        normalized = scaler.fit_transform(data)
    
    return normalized, scaler


def calculate_class_weights(y: np.ndarray) -> Dict[int, float]:
    """
    Calculate class weights for imbalanced datasets
    
    Args:
        y: Target labels
    
    Returns:
        Dictionary mapping class to weight
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    
    return dict(zip(classes, weights))


def split_train_val_test(X: np.ndarray, 
                         y: np.ndarray,
                         train_ratio: float = 0.6,
                         val_ratio: float = 0.2,
                         test_ratio: float = 0.2,
                         shuffle: bool = False,
                         random_state: int = 42) -> Tuple:
    """
    Split data into train, validation, and test sets
    
    Args:
        X: Features
        y: Labels
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        shuffle: Whether to shuffle data
        random_state: Random seed
    
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    n_samples = len(X)
    
    if shuffle:
        indices = np.random.RandomState(random_state).permutation(n_samples)
        X = X[indices]
        y = y[indices]
    
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    
    X_train = X[:train_end]
    X_val = X[train_end:val_end]
    X_test = X[val_end:]
    
    y_train = y[:train_end]
    y_val = y[train_end:val_end]
    y_test = y[val_end:]
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def print_data_info(data: Union[np.ndarray, pd.DataFrame], 
                    name: str = "Data"):
    """
    Print detailed information about data
    
    Args:
        data: Data to analyze (numpy array or DataFrame)
        name: Name to display
    """
    logger.info(f"\n{name} Information:")
    
    if isinstance(data, pd.DataFrame):
        logger.info(f"  Shape: {data.shape}")
        logger.info(f"  Columns: {list(data.columns)}")
        logger.info(f"  Dtypes: {dict(data.dtypes)}")
        logger.info(f"  Missing values: {data.isnull().sum().sum()}")
        logger.info(f"  Memory usage: {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    elif isinstance(data, np.ndarray):
        logger.info(f"  Shape: {data.shape}")
        logger.info(f"  Dtype: {data.dtype}")
        logger.info(f"  Min: {np.min(data):.4f}")
        logger.info(f"  Max: {np.max(data):.4f}")
        logger.info(f"  Mean: {np.mean(data):.4f}")
        logger.info(f"  Std: {np.std(data):.4f}")
        logger.info(f"  Missing values: {np.isnan(data).sum()}")
    
    else:
        logger.warning(f"  Unknown data type: {type(data)}")


def get_gpu_info() -> Dict[str, Any]:
    """
    Get GPU information if available
    
    Returns:
        Dictionary with GPU information
    """
    info = {
        'available': False,
        'device_name': None,
        'device_count': 0,
        'memory_limit': None
    }
    
    try:
        import tensorflow as tf
        
        gpus = tf.config.list_physical_devices('GPU')
        info['available'] = len(gpus) > 0
        info['device_count'] = len(gpus)
        
        if gpus:
            info['device_name'] = gpus[0].name
            
            # Try to get memory info
            try:
                gpu_details = tf.config.experimental.get_device_details(gpus[0])
                info['compute_capability'] = gpu_details.get('compute_capability')
            except:
                pass
    
    except Exception as e:
        logger.warning(f"Could not get GPU info: {e}")
    
    return info


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time string
    
    Args:
        seconds: Time in seconds
    
    Returns:
        Formatted string (e.g., "2h 15m 30s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:
        parts.append(f"{secs}s")
    
    return " ".join(parts)


def create_directory_structure(base_dir: Path, structure: Dict[str, Any]):
    """
    Create directory structure from nested dictionary
    
    Args:
        base_dir: Base directory path
        structure: Nested dictionary representing directory structure
    
    Example:
        structure = {
            'data': {
                'raw': {},
                'processed': {}
            },
            'models': {}
        }
    """
    base_dir = Path(base_dir)
    
    for name, subdirs in structure.items():
        dir_path = base_dir / name
        dir_path.mkdir(parents=True, exist_ok=True)
        
        if isinstance(subdirs, dict) and subdirs:
            create_directory_structure(dir_path, subdirs)


def memory_usage_mb(obj: Any) -> float:
    """
    Get memory usage of object in MB
    
    Args:
        obj: Object to measure
    
    Returns:
        Memory usage in MB
    """
    import sys
    
    if isinstance(obj, pd.DataFrame):
        return obj.memory_usage(deep=True).sum() / 1024**2
    elif isinstance(obj, np.ndarray):
        return obj.nbytes / 1024**2
    else:
        return sys.getsizeof(obj) / 1024**2


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if division by zero
    
    Args:
        a: Numerator
        b: Denominator
        default: Value to return if b is zero
    
    Returns:
        a/b or default
    """
    try:
        return a / b if b != 0 else default
    except (ZeroDivisionError, TypeError):
        return default


def clip_outliers(data: np.ndarray, 
                  lower_percentile: float = 1,
                  upper_percentile: float = 99) -> np.ndarray:
    """
    Clip outliers based on percentiles
    
    Args:
        data: Input array
        lower_percentile: Lower percentile for clipping
        upper_percentile: Upper percentile for clipping
    
    Returns:
        Clipped array
    """
    lower_bound = np.percentile(data, lower_percentile)
    upper_bound = np.percentile(data, upper_percentile)
    
    return np.clip(data, lower_bound, upper_bound)


if __name__ == "__main__":
    # Test helper functions
    print("Testing helper functions...")
    
    # Test random seeds
    set_random_seeds(42)
    
    # Test DataFrame validation
    test_df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [5, 4, 3, 2, 1]
    })
    is_valid, msg = validate_dataframe(test_df, required_columns=['A', 'B'])
    assert is_valid, msg
    
    # Test sequence creation
    data = np.random.rand(100, 5)
    X, y = create_sequences(data, lookback=10, forecast_horizon=1)
    assert X.shape == (91, 10, 5)
    assert y.shape == (91, 1, 5)
    
    # Test normalization
    data = np.random.rand(100, 5)
    normalized, scaler = normalize_data(data, method='minmax')
    assert normalized.min() >= 0 and normalized.max() <= 1
    
    # Test GPU info
    gpu_info = get_gpu_info()
    print(f"GPU Available: {gpu_info['available']}")
    
    # Test time formatting
    assert format_time(3661) == "1h 1m 1s"
    assert format_time(45) == "45s"
    
    print("✓ All helper function tests passed!")
