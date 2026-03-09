"""
============================================================================
Data Loader Module
Handles loading and initial validation of both datasets:
1. water_potability.csv (Classification)
2. water_dataset.mat (Spatio-temporal forecasting)
============================================================================
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Optional, Any
from scipy.io import loadmat

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    WATER_POTABILITY_PATH,
    WATER_DATASET_MAT_PATH,
    CLASSIFICATION_FEATURES,
    FORECASTING_FEATURES
)
from utils.logger import get_logger
from utils.helpers import validate_dataframe, get_missing_value_summary, print_data_info

logger = get_logger()


class WaterPotabilityLoader:
    """
    Loader for water potability classification dataset
    
    Dataset: water_potability.csv
    Features: 9 water quality parameters
    Target: Potability (0/1)
    """
    
    def __init__(self, filepath: Path = WATER_POTABILITY_PATH):
        """
        Initialize loader
        
        Args:
            filepath: Path to CSV file
        """
        self.filepath = Path(filepath)
        self.data = None
        self.X = None
        self.y = None
        self.feature_names = CLASSIFICATION_FEATURES
    
    def load(self, verbose: bool = True) -> Tuple[pd.DataFrame, bool]:
        """
        Load dataset from CSV
        
        Args:
            verbose: Whether to print detailed information
        
        Returns:
            Tuple of (dataframe, success_flag)
        """
        try:
            if verbose:
                logger.info(f"Loading classification dataset: {self.filepath.name}")
            
            # Check file exists
            if not self.filepath.exists():
                logger.error(f"File not found: {self.filepath}")
                return None, False
            
            # Load CSV
            self.data = pd.read_csv(self.filepath)
            
            if verbose:
                logger.success(f"Dataset loaded: {len(self.data)} samples")
            
            # Validate
            is_valid, message = self._validate_data()
            
            if not is_valid:
                logger.error(f"Validation failed: {message}")
                return self.data, False
            
            if verbose:
                self._print_summary()
            
            return self.data, True
        
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return None, False
    
    def _validate_data(self) -> Tuple[bool, str]:
        """
        Validate loaded data
        
        Returns:
            Tuple of (is_valid, message)
        """
        # Check if data loaded
        if self.data is None or len(self.data) == 0:
            return False, "No data loaded"
        
        # Check for target column
        if 'Potability' not in self.data.columns:
            return False, "Missing target column 'Potability'"
        
        # Check feature columns
        expected_features = set(self.feature_names)
        actual_features = set(self.data.columns) - {'Potability'}
        
        if expected_features != actual_features:
            missing = expected_features - actual_features
            extra = actual_features - expected_features
            
            msg = []
            if missing:
                msg.append(f"Missing features: {missing}")
            if extra:
                msg.append(f"Extra features: {extra}")
            
            return False, "; ".join(msg)
        
        # Check target values
        unique_targets = self.data['Potability'].unique()
        if not set(unique_targets).issubset({0, 1}):
            return False, f"Invalid target values: {unique_targets}"
        
        return True, "Valid"
    
    def _print_summary(self):
        """Print dataset summary"""
        logger.subsection("Classification Dataset Summary")
        
        # Basic info
        logger.info(f"  Total samples: {len(self.data)}")
        logger.info(f"  Features: {len(self.feature_names)}")
        logger.info(f"  Feature names: {', '.join(self.feature_names)}")
        
        # Target distribution
        target_counts = self.data['Potability'].value_counts().sort_index()
        logger.info(f"\n  Target Distribution:")
        logger.info(f"    Non-potable (0): {target_counts.get(0, 0)} ({target_counts.get(0, 0)/len(self.data)*100:.1f}%)")
        logger.info(f"    Potable (1): {target_counts.get(1, 0)} ({target_counts.get(1, 0)/len(self.data)*100:.1f}%)")
        
        # Missing values
        missing_summary = get_missing_value_summary(self.data)
        if len(missing_summary) > 0:
            logger.warning(f"\n  Missing Values Detected:")
            for col, row in missing_summary.iterrows():
                logger.warning(f"    {col}: {int(row['Missing_Count'])} ({row['Missing_Percent']:.1f}%)")
        else:
            logger.success("  No missing values")
        
        # Basic statistics
        logger.info(f"\n  Feature Statistics:")
        stats = self.data[self.feature_names].describe()
        logger.info(f"\n{stats.to_string()}")
    
    def get_features_and_target(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Get features (X) and target (y)
        
        Returns:
            Tuple of (X, y)
        """
        if self.data is None:
            logger.error("Data not loaded. Call load() first.")
            return None, None
        
        X = self.data[self.feature_names]
        y = self.data['Potability']
        
        return X, y
    
    def get_feature_names(self) -> list:
        """Get list of feature names"""
        return self.feature_names.copy()


class WaterQualityForecastingLoader:
    """
    Loader for spatio-temporal water quality forecasting dataset
    
    Dataset: water_dataset.mat
    Structure: MATLAB .mat file with spatial-temporal data
    Task: Predict pH values for next day
    """
    
    def __init__(self, filepath: Path = WATER_DATASET_MAT_PATH):
        """
        Initialize loader
        
        Args:
            filepath: Path to .mat file
        """
        self.filepath = Path(filepath)
        self.raw_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.features = None
        self.location_ids = None
        self.location_groups = None
    
    def load(self, verbose: bool = True) -> Tuple[Dict[str, Any], bool]:
        """
        Load dataset from .mat file
        
        Args:
            verbose: Whether to print detailed information
        
        Returns:
            Tuple of (data_dict, success_flag)
        """
        try:
            if verbose:
                logger.info(f"Loading forecasting dataset: {self.filepath.name}")
            
            # Check file exists
            if not self.filepath.exists():
                logger.error(f"File not found: {self.filepath}")
                return None, False
            
            # Load .mat file
            self.raw_data = loadmat(str(self.filepath))
            
            # Extract components
            self._extract_components()
            
            if verbose:
                logger.success("Dataset loaded successfully")
            
            # Validate
            is_valid, message = self._validate_data()
            
            if not is_valid:
                logger.error(f"Validation failed: {message}")
                return None, False
            
            if verbose:
                self._print_summary()
            
            # Return structured data
            data_dict = {
                'X_train': self.X_train,
                'X_test': self.X_test,
                'y_train': self.y_train,
                'y_test': self.y_test,
                'features': self.features,
                'location_ids': self.location_ids,
                'location_groups': self.location_groups
            }
            
            return data_dict, True
        
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            logger.exception("Detailed error:")
            return None, False
    
    def _extract_components(self):
        """Extract components from loaded .mat file"""
        # Extract training data (X_tr is object array of matrices)
        self.X_train = self.raw_data['X_tr'][0]  # Shape: (423,) each element is (37, 11)
        self.X_test = self.raw_data['X_te'][0]   # Shape: (282,) each element is (37, 11)
        
        # Extract target data
        self.y_train = self.raw_data['Y_tr']  # Shape: (37, 423)
        self.y_test = self.raw_data['Y_te']   # Shape: (37, 282)
        
        # Extract metadata
        self.features = [str(f[0]) for f in self.raw_data['features'][0]]
        self.location_ids = self.raw_data['location_ids'].flatten()
        self.location_groups = self.raw_data['location_group'][0]
    
    def _validate_data(self) -> Tuple[bool, str]:
        """
        Validate loaded data
        
        Returns:
            Tuple of (is_valid, message)
        """
        # Check if data loaded
        if self.X_train is None or self.X_test is None:
            return False, "Data not properly extracted"
        
        # Check dimensions
        n_train = len(self.X_train)
        n_test = len(self.X_test)
        
        if n_train == 0 or n_test == 0:
            return False, "Empty train or test set"
        
        # Check feature consistency
        if len(self.features) != 11:
            return False, f"Expected 11 features, got {len(self.features)}"
        
        # Check location consistency
        n_locations = len(self.location_ids)
        if n_locations != 37:
            return False, f"Expected 37 locations, got {n_locations}"
        
        # Check y dimensions match X
        if self.y_train.shape[1] != n_train:
            return False, f"y_train timesteps ({self.y_train.shape[1]}) != X_train ({n_train})"
        
        if self.y_test.shape[1] != n_test:
            return False, f"y_test timesteps ({self.y_test.shape[1]}) != X_test ({n_test})"
        
        return True, "Valid"
    
    def _print_summary(self):
        """Print dataset summary"""
        logger.subsection("Forecasting Dataset Summary")
        
        # Basic info
        n_train = len(self.X_train)
        n_test = len(self.X_test)
        n_total = n_train + n_test
        
        logger.info(f"  Total days: {n_total}")
        logger.info(f"  Training days: {n_train} ({n_train/n_total*100:.1f}%)")
        logger.info(f"  Testing days: {n_test} ({n_test/n_total*100:.1f}%)")
        logger.info(f"  Locations: {len(self.location_ids)}")
        logger.info(f"  Features per location: {len(self.features)}")
        
        # Feature names
        logger.info(f"\n  Features:")
        for i, feat in enumerate(self.features, 1):
            logger.info(f"    {i}. {feat}")
        
        # Spatial structure
        logger.info(f"\n  Spatial Structure:")
        logger.info(f"    Location groups: {len(self.location_groups)}")
        for i, group in enumerate(self.location_groups, 1):
            logger.info(f"      Group {i}: {len(group[0])} locations")
        
        # Sample data shape
        logger.info(f"\n  Data Structure:")
        logger.info(f"    X_train[0] shape: {self.X_train[0].shape} (locations × features)")
        logger.info(f"    y_train shape: {self.y_train.shape} (locations × days)")
        
        # Target statistics
        y_train_flat = self.y_train.flatten()
        logger.info(f"\n  Target (pH) Statistics:")
        logger.info(f"    Min: {np.min(y_train_flat):.4f}")
        logger.info(f"    Max: {np.max(y_train_flat):.4f}")
        logger.info(f"    Mean: {np.mean(y_train_flat):.4f}")
        logger.info(f"    Std: {np.std(y_train_flat):.4f}")
    
    def get_train_test_split(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get train-test split
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        if self.X_train is None:
            logger.error("Data not loaded. Call load() first.")
            return None, None, None, None
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def get_feature_names(self) -> list:
        """Get list of feature names"""
        return self.features.copy() if self.features else []
    
    def get_location_info(self) -> Dict[str, Any]:
        """
        Get location information
        
        Returns:
            Dictionary with location metadata
        """
        return {
            'location_ids': self.location_ids,
            'n_locations': len(self.location_ids),
            'location_groups': self.location_groups,
            'n_groups': len(self.location_groups)
        }


def load_all_datasets(verbose: bool = True) -> Dict[str, Any]:
    """
    Load all datasets for the project
    
    Args:
        verbose: Whether to print detailed information
    
    Returns:
        Dictionary containing all loaded datasets
    """
    if verbose:
        logger.section("LOADING ALL DATASETS")
    
    results = {
        'classification': None,
        'forecasting': None,
        'success': False
    }
    
    # Load classification dataset
    if verbose:
        logger.info("\n1. Loading Classification Dataset...")
    
    potability_loader = WaterPotabilityLoader()
    potability_data, success1 = potability_loader.load(verbose=verbose)
    
    if success1:
        X_class, y_class = potability_loader.get_features_and_target()
        results['classification'] = {
            'data': potability_data,
            'X': X_class,
            'y': y_class,
            'feature_names': potability_loader.get_feature_names(),
            'loader': potability_loader
        }
    
    # Load forecasting dataset
    if verbose:
        logger.info("\n2. Loading Forecasting Dataset...")
    
    forecast_loader = WaterQualityForecastingLoader()
    forecast_data, success2 = forecast_loader.load(verbose=verbose)
    
    if success2:
        results['forecasting'] = {
            'data': forecast_data,
            'X_train': forecast_data['X_train'],
            'X_test': forecast_data['X_test'],
            'y_train': forecast_data['y_train'],
            'y_test': forecast_data['y_test'],
            'features': forecast_data['features'],
            'location_info': forecast_loader.get_location_info(),
            'loader': forecast_loader
        }
    
    # Overall success
    results['success'] = success1 and success2
    
    if verbose:
        logger.section("DATASET LOADING COMPLETE")
        if results['success']:
            logger.success("All datasets loaded successfully!")
        else:
            logger.error("Some datasets failed to load")
    
    return results


if __name__ == "__main__":
    # Test data loaders
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from src.utils.logger import get_logger
    from src.config import LOGS_DIR
    
    # Initialize logger
    logger = get_logger(log_dir=LOGS_DIR)
    
    # Test loading all datasets
    datasets = load_all_datasets(verbose=True)
    
    if datasets['success']:
        print("\n" + "="*70)
        print("✓ All data loaders working correctly!")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("✗ Some data loaders failed!")
        print("="*70)
        sys.exit(1)
