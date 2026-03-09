"""
============================================================================
Forecasting Data Preprocessing Pipeline
Handles water quality forecasting dataset preprocessing including:
- Data normalization
- Sequence generation
- Train-validation-test splitting
- Spatial data handling
============================================================================
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    FORECASTING_CONFIG,
    PROCESSED_DATA_DIR,
    MODEL_PATHS,
    RANDOM_SEED
)
from utils.logger import get_logger
from utils.file_handler import FileHandler
from utils.helpers import set_random_seeds
from data.sequence_generator import SequenceGenerator

logger = get_logger()


class ForecastingPreprocessor:
    """
    Preprocessing pipeline for water quality forecasting
    
    Steps:
    1. Normalize features
    2. Create sequences
    3. Split into train/val/test
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize preprocessor
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or FORECASTING_CONFIG
        self.feature_scaler = None
        self.target_scaler = None
        self.sequence_generator = None
        
        # Statistics
        self.stats = {
            'original_timesteps': 0,
            'n_locations': 0,
            'n_features': 0,
            'sequences_created': 0,
            'train_sequences': 0,
            'val_sequences': 0,
            'test_sequences': 0
        }
        
        set_random_seeds(RANDOM_SEED)
    
    def fit_transform_train(self,
                           X_train: np.ndarray,
                           y_train: np.ndarray,
                           verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit preprocessor on training data and transform
        
        Args:
            X_train: Training features (timesteps, locations, features)
                    or list of (locations, features) matrices
            y_train: Training targets (locations, timesteps)
            verbose: Print progress
        
        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        if verbose:
            logger.section("FORECASTING PREPROCESSING (TRAIN)")
        
        # Convert object array to 3D numpy array if needed
        if isinstance(X_train, np.ndarray) and X_train.dtype == object:
            n_timesteps = len(X_train)
            X_stacked = np.stack([X_train[i] for i in range(n_timesteps)], axis=0)
        else:
            X_stacked = X_train
        
        self.stats['original_timesteps'] = X_stacked.shape[0]
        self.stats['n_locations'] = X_stacked.shape[1]
        self.stats['n_features'] = X_stacked.shape[2]
        
        if verbose:
            logger.info(f"  Original shape: {X_stacked.shape}")
            logger.info(f"    Timesteps: {self.stats['original_timesteps']}")
            logger.info(f"    Locations: {self.stats['n_locations']}")
            logger.info(f"    Features: {self.stats['n_features']}")
        
        # Step 1: Normalize features
        X_normalized = self._normalize_features(X_stacked, fit=True, verbose=verbose)
        
        # Step 1.5: Normalize targets (NEW!)
        y_normalized = self._normalize_target(y_train, fit=True)
        
        if verbose:
            logger.info(f"  Target normalized: [{y_normalized.min():.4f}, {y_normalized.max():.4f}]")
        
        # Step 2: Create sequences
        X_sequences, y_sequences = self._create_sequences(
            X_normalized, y_normalized, verbose=verbose
        )
        
        self.stats['sequences_created'] = len(X_sequences)
        
        if verbose:
            self._print_summary()
        
        return X_sequences, y_sequences
    
    def transform_test(self,
                      X_test: np.ndarray,
                      y_test: np.ndarray,
                      verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform test data using fitted preprocessor
        
        Args:
            X_test: Test features
            y_test: Test targets
            verbose: Print progress
        
        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        if verbose:
            logger.section("FORECASTING PREPROCESSING (TEST)")
        
        if self.feature_scaler is None:
            raise ValueError("Preprocessor not fitted. Call fit_transform_train first.")
        
        # Convert object array if needed
        if isinstance(X_test, np.ndarray) and X_test.dtype == object:
            n_timesteps = len(X_test)
            X_stacked = np.stack([X_test[i] for i in range(n_timesteps)], axis=0)
        else:
            X_stacked = X_test
        
        # Normalize using fitted scaler
        X_normalized = self._normalize_features(X_stacked, fit=False, verbose=verbose)
        
        # Normalize targets (NEW!)
        y_normalized = self._normalize_target(y_test, fit=False)
        
        # Create sequences
        X_sequences, y_sequences = self._create_sequences(
            X_normalized, y_normalized, verbose=verbose
        )
        
        if verbose:
            logger.info(f"  Test sequences created: {len(X_sequences)}")
        
        return X_sequences, y_sequences
    
    def _normalize_features(self,
                           X: np.ndarray,
                           fit: bool = True,
                           verbose: bool = False) -> np.ndarray:
        """
        Normalize features
        
        Args:
            X: Features (timesteps, locations, features)
            fit: Whether to fit scaler
            verbose: Print progress
        
        Returns:
            Normalized features
        """
        if fit:
            if verbose:
                logger.subsection("Step 1: Normalizing Features")
            
            method = self.config.get('scaling_method', 'minmax')
            feature_range = self.config.get('feature_range', (0, 1))
            
            if method == 'minmax':
                self.feature_scaler = MinMaxScaler(feature_range=feature_range)
                scaler_name = "MinMaxScaler"
            elif method == 'standard':
                self.feature_scaler = StandardScaler()
                scaler_name = "StandardScaler"
            else:
                raise ValueError(f"Unknown scaling method: {method}")
            
            if verbose:
                logger.info(f"  Using {scaler_name}")
        
        # Reshape to 2D for scaling
        original_shape = X.shape
        X_reshaped = X.reshape(-1, X.shape[-1])
        
        # Scale
        if fit:
            X_scaled = self.feature_scaler.fit_transform(X_reshaped)
        else:
            X_scaled = self.feature_scaler.transform(X_reshaped)
        
        # Reshape back
        X_normalized = X_scaled.reshape(original_shape)
        
        if fit and verbose:
            logger.success("  Features normalized")
            logger.info(f"  Range: [{X_normalized.min():.4f}, {X_normalized.max():.4f}]")
        
        return X_normalized
    
    def _normalize_target(self, y: np.ndarray, fit: bool = False) -> np.ndarray:
        """
        Normalize target variable (pH)
        
        Args:
            y: Target values (locations, timesteps)
            fit: Whether to fit scaler
        
        Returns:
            Normalized targets (same shape as input)
        """
        # Save original shape
        original_shape = y.shape
        
        if fit:
            self.target_scaler = MinMaxScaler(feature_range=(0, 1))
            y_scaled = self.target_scaler.fit_transform(y.reshape(-1, 1))
        else:
            if self.target_scaler is None:
                raise ValueError("Target scaler not fitted!")
            y_scaled = self.target_scaler.transform(y.reshape(-1, 1))
        
        # Reshape back to original shape instead of flattening
        return y_scaled.reshape(original_shape)
    
    def inverse_transform_target(self, y: np.ndarray) -> np.ndarray:
        """
        Inverse transform target back to original scale
        
        Args:
            y: Normalized target values (any shape)
        
        Returns:
            Original scale targets (same shape as input)
        """
        if self.target_scaler is None:
            raise ValueError("Target scaler not fitted!")
        
        # Save original shape
        original_shape = y.shape
        
        # Inverse transform
        y_original = self.target_scaler.inverse_transform(y.reshape(-1, 1))
        
        # Reshape back to original shape
        return y_original.reshape(original_shape)
    
    def _create_sequences(self,
                         X: np.ndarray,
                         y: np.ndarray,
                         verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training
        
        Args:
            X: Normalized features (timesteps, locations, features)
            y: Targets (locations, timesteps)
            verbose: Print progress
        
        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        if self.sequence_generator is None:
            lookback = self.config.get('lookback_window', 30)
            forecast_horizon = self.config.get('forecast_horizon', 1)
            stride = self.config.get('stride', 1)
            
            self.sequence_generator = SequenceGenerator(
                lookback=lookback,
                forecast_horizon=forecast_horizon,
                stride=stride
            )
        
        # Create sequences for all locations
        X_sequences, y_sequences = self.sequence_generator.create_multi_location_sequences(
            X, y, n_locations=X.shape[1], verbose=verbose
        )
        
        return X_sequences, y_sequences
    
    def split_sequences(self,
                       X_sequences: np.ndarray,
                       y_sequences: np.ndarray,
                       train_ratio: float = None,
                       val_ratio: float = None,
                       verbose: bool = True) -> Tuple:
        """
        Split sequences into train, validation, and test sets
        
        Args:
            X_sequences: Input sequences
            y_sequences: Target sequences
            train_ratio: Training set proportion
            val_ratio: Validation set proportion
            verbose: Print progress
        
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        train_ratio = train_ratio or self.config.get('train_ratio', 0.6)
        val_ratio = val_ratio or self.config.get('val_ratio', 0.2)
        test_ratio = 1.0 - train_ratio - val_ratio
        
        n_sequences = len(X_sequences)
        
        # Calculate split indices
        train_end = int(n_sequences * train_ratio)
        val_end = int(n_sequences * (train_ratio + val_ratio))
        
        # Split (no shuffling for time-series data)
        X_train = X_sequences[:train_end]
        X_val = X_sequences[train_end:val_end]
        X_test = X_sequences[val_end:]
        
        y_train = y_sequences[:train_end]
        y_val = y_sequences[train_end:val_end]
        y_test = y_sequences[val_end:]
        
        self.stats['train_sequences'] = len(X_train)
        self.stats['val_sequences'] = len(X_val)
        self.stats['test_sequences'] = len(X_test)
        
        if verbose:
            logger.subsection("Sequence Splitting")
            logger.info(f"  Train: {len(X_train)} ({train_ratio*100:.1f}%)")
            logger.info(f"  Validation: {len(X_val)} ({val_ratio*100:.1f}%)")
            logger.info(f"  Test: {len(X_test)} ({test_ratio*100:.1f}%)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def inverse_transform_predictions(self,
                                     predictions: np.ndarray) -> np.ndarray:
        """
        Inverse transform predictions to original scale
        
        Args:
            predictions: Scaled predictions
        
        Returns:
            Original scale predictions
        """
        # Note: Targets (pH) are not scaled in original dataset
        # They're already normalized (0-1 range)
        # So we just return as is
        return predictions
    
    def _print_summary(self):
        """Print preprocessing summary"""
        logger.section("PREPROCESSING SUMMARY")
        
        logger.info(f"  Original timesteps: {self.stats['original_timesteps']}")
        logger.info(f"  Locations: {self.stats['n_locations']}")
        logger.info(f"  Features per location: {self.stats['n_features']}")
        logger.info(f"  Sequences created: {self.stats['sequences_created']}")
        
        if self.sequence_generator:
            info = self.sequence_generator.get_sequence_info()
            logger.info(f"\n  Sequence Configuration:")
            logger.info(f"    Lookback: {info['lookback']}")
            logger.info(f"    Forecast horizon: {info['forecast_horizon']}")
            logger.info(f"    Input shape: {info['input_shape']}")
            logger.info(f"    Output shape: {info['output_shape']}")
    
    def save(self, output_dir: Path = PROCESSED_DATA_DIR):
        """
        Save preprocessor components
        
        Args:
            output_dir: Directory to save components
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save scaler
        if self.feature_scaler is not None:
            FileHandler.save_model_joblib(
                self.feature_scaler,
                MODEL_PATHS['scaler_forecasting']
            )
        
        # Save target scaler (NEW!)
        if self.target_scaler is not None:
            FileHandler.save_model_joblib(
                self.target_scaler,
                output_dir / 'target_scaler.pkl'
            )
        
        # Save sequence generator config
        if self.sequence_generator is not None:
            seq_info = self.sequence_generator.get_sequence_info()
            FileHandler.save_json(
                seq_info,
                output_dir / 'sequence_config.json'
            )
        
        # Save statistics
        FileHandler.save_json(
            self.stats,
            output_dir / 'forecasting_preprocessing_stats.json'
        )
        
        logger.success("Forecasting preprocessor saved successfully")


def preprocess_forecasting_data(X_train: np.ndarray,
                               X_test: np.ndarray,
                               y_train: np.ndarray,
                               y_test: np.ndarray,
                               save_results: bool = True,
                               verbose: bool = True) -> Dict[str, Any]:
    """
    Complete preprocessing pipeline for forecasting data
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training targets
        y_test: Test targets
        save_results: Whether to save preprocessed data
        verbose: Print progress
    
    Returns:
        Dictionary containing all preprocessed components
    """
    # Initialize preprocessor
    preprocessor = ForecastingPreprocessor()
    
    # Process training data
    X_train_seq, y_train_seq = preprocessor.fit_transform_train(
        X_train, y_train, verbose=verbose
    )
    
    # Split training sequences into train/val
    X_train_split, X_val, X_test_placeholder, y_train_split, y_val, y_test_placeholder = \
        preprocessor.split_sequences(X_train_seq, y_train_seq, verbose=verbose)
    
    # Process test data
    X_test_seq, y_test_seq = preprocessor.transform_test(
        X_test, y_test, verbose=verbose
    )
    
    # Save results
    if save_results:
        logger.subsection("Saving Preprocessed Sequences")
        
        FileHandler.save_numpy(X_train_split, PROCESSED_DATA_DIR / 'forecasting_X_train.npy')
        FileHandler.save_numpy(X_val, PROCESSED_DATA_DIR / 'forecasting_X_val.npy')
        FileHandler.save_numpy(X_test_seq, PROCESSED_DATA_DIR / 'forecasting_X_test.npy')
        
        FileHandler.save_numpy(y_train_split, PROCESSED_DATA_DIR / 'forecasting_y_train.npy')
        FileHandler.save_numpy(y_val, PROCESSED_DATA_DIR / 'forecasting_y_val.npy')
        FileHandler.save_numpy(y_test_seq, PROCESSED_DATA_DIR / 'forecasting_y_test.npy')
        
        preprocessor.save()
    
    return {
        'X_train': X_train_split,
        'X_val': X_val,
        'X_test': X_test_seq,
        'y_train': y_train_split,
        'y_val': y_val,
        'y_test': y_test_seq,
        'preprocessor': preprocessor,
        'scaler': preprocessor.feature_scaler,
        'sequence_generator': preprocessor.sequence_generator,
        'stats': preprocessor.stats
    }


if __name__ == "__main__":
    # Test forecasting preprocessing
    from data.data_loader import WaterQualityForecastingLoader
    
    # Load data
    loader = WaterQualityForecastingLoader()
    data, success = loader.load(verbose=False)
    
    if success:
        X_train = data['X_train']
        X_test = data['X_test']
        y_train = data['y_train']
        y_test = data['y_test']
        
        # Preprocess
        results = preprocess_forecasting_data(
            X_train, X_test, y_train, y_test,
            verbose=True
        )
        
        print("\n" + "="*70)
        print("✓ Forecasting preprocessing complete!")
        print(f"  Train sequences: {len(results['X_train'])}")
        print(f"  Validation sequences: {len(results['X_val'])}")
        print(f"  Test sequences: {len(results['X_test'])}")
        print(f"  Input shape: {results['X_train'].shape}")
        print(f"  Output shape: {results['y_train'].shape}")
        print("="*70)
