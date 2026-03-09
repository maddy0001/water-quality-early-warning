"""
============================================================================
File Handler Module
Handles all file I/O operations including model saving/loading, 
data persistence, and JSON operations
============================================================================
"""

import json
import pickle
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from .logger import get_logger

logger = get_logger()


class FileHandler:
    """
    Centralized file handling operations
    
    Supports:
    - Model saving/loading (pickle, joblib, h5)
    - Data saving/loading (CSV, JSON, numpy)
    - Metrics persistence
    - Directory management
    """
    
    @staticmethod
    def save_model_pickle(model: Any, filepath: Path) -> bool:
        """
        Save model using pickle
        
        Args:
            model: Model object to save
            filepath: Path to save the model
        
        Returns:
            True if successful, False otherwise
        """
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
            
            logger.success(f"Model saved: {filepath.name}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False
    
    @staticmethod
    def load_model_pickle(filepath: Path) -> Optional[Any]:
        """
        Load model from pickle file
        
        Args:
            filepath: Path to the model file
        
        Returns:
            Loaded model or None if failed
        """
        try:
            filepath = Path(filepath)
            
            if not filepath.exists():
                logger.error(f"Model file not found: {filepath}")
                return None
            
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
            
            logger.success(f"Model loaded: {filepath.name}")
            return model
        
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None
    
    @staticmethod
    def save_model_joblib(model: Any, filepath: Path) -> bool:
        """
        Save model using joblib (better for large numpy arrays)
        
        Args:
            model: Model object to save
            filepath: Path to save the model
        
        Returns:
            True if successful, False otherwise
        """
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            joblib.dump(model, filepath)
            
            logger.success(f"Model saved (joblib): {filepath.name}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False
    
    @staticmethod
    def load_model_joblib(filepath: Path) -> Optional[Any]:
        """
        Load model from joblib file
        
        Args:
            filepath: Path to the model file
        
        Returns:
            Loaded model or None if failed
        """
        try:
            filepath = Path(filepath)
            
            if not filepath.exists():
                logger.error(f"Model file not found: {filepath}")
                return None
            
            model = joblib.load(filepath)
            
            logger.success(f"Model loaded (joblib): {filepath.name}")
            return model
        
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None
    
    @staticmethod
    def save_keras_model(model: Any, filepath: Path) -> bool:
        """
        Save Keras/TensorFlow model
        
        Args:
            model: Keras model object
            filepath: Path to save the model (.h5 or .keras)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            model.save(filepath)
            
            logger.success(f"Keras model saved: {filepath.name}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to save Keras model: {e}")
            return False
    
    @staticmethod
    def load_keras_model(filepath: Path) -> Optional[Any]:
        """
        Load Keras/TensorFlow model
        
        Args:
            filepath: Path to the model file
        
        Returns:
            Loaded model or None if failed
        """
        try:
            from tensorflow import keras
            
            filepath = Path(filepath)
            
            if not filepath.exists():
                logger.error(f"Model file not found: {filepath}")
                return None
            
            model = keras.models.load_model(filepath)
            
            logger.success(f"Keras model loaded: {filepath.name}")
            return model
        
        except Exception as e:
            logger.error(f"Failed to load Keras model: {e}")
            return None
    
    @staticmethod
    def save_numpy(array: np.ndarray, filepath: Path) -> bool:
        """
        Save numpy array
        
        Args:
            array: Numpy array to save
            filepath: Path to save the array
        
        Returns:
            True if successful, False otherwise
        """
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            np.save(filepath, array)
            
            logger.success(f"Numpy array saved: {filepath.name}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to save numpy array: {e}")
            return False
    
    @staticmethod
    def load_numpy(filepath: Path) -> Optional[np.ndarray]:
        """
        Load numpy array
        
        Args:
            filepath: Path to the numpy file
        
        Returns:
            Loaded array or None if failed
        """
        try:
            filepath = Path(filepath)
            
            if not filepath.exists():
                logger.error(f"Numpy file not found: {filepath}")
                return None
            
            array = np.load(filepath, allow_pickle=True)
            
            logger.success(f"Numpy array loaded: {filepath.name}")
            return array
        
        except Exception as e:
            logger.error(f"Failed to load numpy array: {e}")
            return None
    
    @staticmethod
    def save_dataframe(df: pd.DataFrame, filepath: Path, index: bool = False) -> bool:
        """
        Save pandas DataFrame to CSV
        
        Args:
            df: DataFrame to save
            filepath: Path to save the CSV
            index: Whether to save index
        
        Returns:
            True if successful, False otherwise
        """
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            df.to_csv(filepath, index=index)
            
            logger.success(f"DataFrame saved: {filepath.name} ({len(df)} rows)")
            return True
        
        except Exception as e:
            logger.error(f"Failed to save DataFrame: {e}")
            return False
    
    @staticmethod
    def load_dataframe(filepath: Path) -> Optional[pd.DataFrame]:
        """
        Load pandas DataFrame from CSV
        
        Args:
            filepath: Path to the CSV file
        
        Returns:
            Loaded DataFrame or None if failed
        """
        try:
            filepath = Path(filepath)
            
            if not filepath.exists():
                logger.error(f"CSV file not found: {filepath}")
                return None
            
            df = pd.read_csv(filepath)
            
            logger.success(f"DataFrame loaded: {filepath.name} ({len(df)} rows)")
            return df
        
        except Exception as e:
            logger.error(f"Failed to load DataFrame: {e}")
            return None
    
    @staticmethod
    def save_json(data: Union[Dict, List], filepath: Path, indent: int = 2) -> bool:
        """
        Save data to JSON file
        
        Args:
            data: Dictionary or list to save
            filepath: Path to save the JSON
            indent: JSON indentation level
        
        Returns:
            True if successful, False otherwise
        """
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=indent, default=str)
            
            logger.success(f"JSON saved: {filepath.name}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to save JSON: {e}")
            return False
    
    @staticmethod
    def load_json(filepath: Path) -> Optional[Union[Dict, List]]:
        """
        Load data from JSON file
        
        Args:
            filepath: Path to the JSON file
        
        Returns:
            Loaded data or None if failed
        """
        try:
            filepath = Path(filepath)
            
            if not filepath.exists():
                logger.error(f"JSON file not found: {filepath}")
                return None
            
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            logger.success(f"JSON loaded: {filepath.name}")
            return data
        
        except Exception as e:
            logger.error(f"Failed to load JSON: {e}")
            return None
    
    @staticmethod
    def save_metrics(metrics: Dict[str, Any], filepath: Path, 
                    timestamp: bool = True) -> bool:
        """
        Save metrics dictionary to JSON with optional timestamp
        
        Args:
            metrics: Dictionary of metrics
            filepath: Path to save the metrics
            timestamp: Whether to add timestamp
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if timestamp:
                metrics['timestamp'] = datetime.now().isoformat()
            
            return FileHandler.save_json(metrics, filepath)
        
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
            return False
    
    @staticmethod
    def create_directory(dirpath: Path) -> bool:
        """
        Create directory if it doesn't exist
        
        Args:
            dirpath: Directory path to create
        
        Returns:
            True if successful, False otherwise
        """
        try:
            dirpath = Path(dirpath)
            dirpath.mkdir(parents=True, exist_ok=True)
            return True
        
        except Exception as e:
            logger.error(f"Failed to create directory: {e}")
            return False
    
    @staticmethod
    def file_exists(filepath: Path) -> bool:
        """
        Check if file exists
        
        Args:
            filepath: Path to check
        
        Returns:
            True if exists, False otherwise
        """
        return Path(filepath).exists()
    
    @staticmethod
    def get_file_size(filepath: Path) -> Optional[int]:
        """
        Get file size in bytes
        
        Args:
            filepath: Path to the file
        
        Returns:
            File size in bytes or None if file doesn't exist
        """
        try:
            filepath = Path(filepath)
            if filepath.exists():
                return filepath.stat().st_size
            return None
        
        except Exception as e:
            logger.error(f"Failed to get file size: {e}")
            return None
    
    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """
        Format file size in human-readable format
        
        Args:
            size_bytes: Size in bytes
        
        Returns:
            Formatted string (e.g., "1.5 MB")
        """
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} PB"


# Convenience functions
def save_model(model: Any, filepath: Path, method: str = 'joblib') -> bool:
    """
    Save model using specified method
    
    Args:
        model: Model to save
        filepath: Path to save
        method: 'pickle', 'joblib', or 'keras'
    
    Returns:
        True if successful
    """
    if method == 'pickle':
        return FileHandler.save_model_pickle(model, filepath)
    elif method == 'joblib':
        return FileHandler.save_model_joblib(model, filepath)
    elif method == 'keras':
        return FileHandler.save_keras_model(model, filepath)
    else:
        logger.error(f"Unknown save method: {method}")
        return False


def load_model(filepath: Path, method: str = 'joblib') -> Optional[Any]:
    """
    Load model using specified method
    
    Args:
        filepath: Path to model
        method: 'pickle', 'joblib', or 'keras'
    
    Returns:
        Loaded model or None
    """
    if method == 'pickle':
        return FileHandler.load_model_pickle(filepath)
    elif method == 'joblib':
        return FileHandler.load_model_joblib(filepath)
    elif method == 'keras':
        return FileHandler.load_keras_model(filepath)
    else:
        logger.error(f"Unknown load method: {method}")
        return None


if __name__ == "__main__":
    # Test file handler
    import tempfile
    import numpy as np
    import pandas as pd
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Test numpy save/load
        test_array = np.random.rand(100, 10)
        FileHandler.save_numpy(test_array, temp_path / "test.npy")
        loaded_array = FileHandler.load_numpy(temp_path / "test.npy")
        assert np.allclose(test_array, loaded_array)
        
        # Test DataFrame save/load
        test_df = pd.DataFrame(np.random.rand(50, 5))
        FileHandler.save_dataframe(test_df, temp_path / "test.csv")
        loaded_df = FileHandler.load_dataframe(temp_path / "test.csv")
        assert test_df.shape == loaded_df.shape
        
        # Test JSON save/load
        test_dict = {"accuracy": 0.85, "loss": 0.23, "epochs": 100}
        FileHandler.save_json(test_dict, temp_path / "test.json")
        loaded_dict = FileHandler.load_json(temp_path / "test.json")
        assert test_dict == loaded_dict
        
        print("✓ All file handler tests passed!")
