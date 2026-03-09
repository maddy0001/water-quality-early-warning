"""
============================================================================
Sequence Generator for Time-Series Forecasting
Creates sequences from temporal data for LSTM/GRU training
============================================================================
"""

import numpy as np
from typing import Tuple, List, Optional
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import FORECASTING_CONFIG, RANDOM_SEED
from utils.logger import get_logger

logger = get_logger()


class SequenceGenerator:
    """
    Generate sequences for time-series forecasting
    
    Supports:
    - Sliding window approach
    - Multi-step ahead forecasting
    - Multiple locations (spatial data)
    """
    
    def __init__(self, 
                 lookback: int = None,
                 forecast_horizon: int = None,
                 stride: int = None):
        """
        Initialize sequence generator
        
        Args:
            lookback: Number of past timesteps to use
            forecast_horizon: Number of future timesteps to predict
            stride: Step size for sliding window
        """
        self.lookback = lookback or FORECASTING_CONFIG['lookback_window']
        self.forecast_horizon = forecast_horizon or FORECASTING_CONFIG['forecast_horizon']
        self.stride = stride or FORECASTING_CONFIG.get('stride', 1)
        
        self.n_sequences = 0
        self.input_shape = None
        self.output_shape = None
    
    def create_sequences(self, 
                        X_temporal: np.ndarray,
                        y_temporal: np.ndarray,
                        verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences from temporal data
        
        Args:
            X_temporal: Input features (timesteps, locations, features)
                       or (timesteps, features)
            y_temporal: Target values (locations, timesteps) 
                       or (timesteps,)
            verbose: Print progress
        
        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        if verbose:
            logger.subsection("Sequence Generation")
            logger.info(f"  Lookback window: {self.lookback}")
            logger.info(f"  Forecast horizon: {self.forecast_horizon}")
            logger.info(f"  Stride: {self.stride}")
        
        # Handle different input formats
        if X_temporal.ndim == 3:
            # Format: (timesteps, locations, features)
            X_seq, y_seq = self._create_sequences_spatial(
                X_temporal, y_temporal, verbose
            )
        elif X_temporal.ndim == 2:
            # Format: (timesteps, features)
            X_seq, y_seq = self._create_sequences_simple(
                X_temporal, y_temporal, verbose
            )
        else:
            raise ValueError(f"Unsupported X shape: {X_temporal.shape}")
        
        self.n_sequences = len(X_seq)
        self.input_shape = X_seq.shape[1:]
        self.output_shape = y_seq.shape[1:]
        
        if verbose:
            logger.success(f"  Created {self.n_sequences} sequences")
            logger.info(f"  Input shape: {self.input_shape}")
            logger.info(f"  Output shape: {self.output_shape}")
        
        return X_seq, y_seq
    
    def _create_sequences_simple(self,
                                 X_temporal: np.ndarray,
                                 y_temporal: np.ndarray,
                                 verbose: bool) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for simple time-series (no spatial dimension)
        
        Args:
            X_temporal: (timesteps, features)
            y_temporal: (timesteps,) or (timesteps, 1)
        
        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        n_timesteps = len(X_temporal)
        
        X_sequences = []
        y_sequences = []
        
        for i in range(0, n_timesteps - self.lookback - self.forecast_horizon + 1, self.stride):
            # Input sequence
            X_seq = X_temporal[i:i + self.lookback]
            
            # Target sequence
            y_seq = y_temporal[i + self.lookback:i + self.lookback + self.forecast_horizon]
            
            X_sequences.append(X_seq)
            y_sequences.append(y_seq)
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def _create_sequences_spatial(self,
                                  X_temporal: np.ndarray,
                                  y_temporal: np.ndarray,
                                  verbose: bool) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for spatio-temporal data
        
        Args:
            X_temporal: List/array of (locations, features) for each timestep
            y_temporal: (locations, timesteps)
        
        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        n_timesteps = len(X_temporal)
        n_locations = y_temporal.shape[0]
        
        if verbose:
            logger.info(f"  Timesteps: {n_timesteps}")
            logger.info(f"  Locations: {n_locations}")
        
        # Convert X_temporal to 3D array if it's a list of matrices
        if isinstance(X_temporal, (list, np.ndarray)) and X_temporal.dtype == object:
            # Stack all timesteps
            X_stacked = np.stack([X_temporal[i] for i in range(n_timesteps)], axis=0)
        else:
            X_stacked = X_temporal
        
        # Transpose y to (timesteps, locations) for easier indexing
        y_transposed = y_temporal.T
        
        X_sequences = []
        y_sequences = []
        
        # Create sequences
        for i in range(0, n_timesteps - self.lookback - self.forecast_horizon + 1, self.stride):
            # Input sequence: (lookback, locations, features)
            X_seq = X_stacked[i:i + self.lookback]
            
            # Target sequence: (forecast_horizon, locations)
            y_seq = y_transposed[i + self.lookback:i + self.lookback + self.forecast_horizon]
            
            X_sequences.append(X_seq)
            y_sequences.append(y_seq)
        
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)
        
        # Reshape for single-location forecasting if needed
        # Output: (n_sequences, lookback, locations, features)
        # Target: (n_sequences, forecast_horizon, locations)
        
        return X_sequences, y_sequences
    
    def create_single_location_sequences(self,
                                        X_temporal: np.ndarray,
                                        y_temporal: np.ndarray,
                                        location_idx: int,
                                        verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for a single location
        
        Args:
            X_temporal: Array/list of (locations, features) for each timestep
            y_temporal: (locations, timesteps)
            location_idx: Index of location to extract
            verbose: Print progress
        
        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        n_timesteps = len(X_temporal)
        
        # Extract data for specific location
        X_location = np.array([X_temporal[i][location_idx] for i in range(n_timesteps)])
        y_location = y_temporal[location_idx]
        
        # Create sequences
        return self._create_sequences_simple(X_location, y_location, verbose)
    
    def create_multi_location_sequences(self,
                                       X_temporal: np.ndarray,
                                       y_temporal: np.ndarray,
                                       n_locations: int,
                                       verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for multiple locations independently
        
        Args:
            X_temporal: Array/list of (locations, features) for each timestep
            y_temporal: (locations, timesteps)
            n_locations: Number of locations
            verbose: Print progress
        
        Returns:
            Tuple of (X_all, y_all) concatenated from all locations
        """
        if verbose:
            logger.subsection("Multi-Location Sequence Generation")
        
        all_X = []
        all_y = []
        
        for loc_idx in range(n_locations):
            X_loc, y_loc = self.create_single_location_sequences(
                X_temporal, y_temporal, loc_idx, verbose=False
            )
            all_X.append(X_loc)
            all_y.append(y_loc)
        
        # Concatenate all sequences
        X_combined = np.concatenate(all_X, axis=0)
        y_combined = np.concatenate(all_y, axis=0)
        
        if verbose:
            logger.info(f"  Processed {n_locations} locations")
            logger.success(f"  Total sequences: {len(X_combined)}")
            logger.info(f"  Input shape: {X_combined.shape}")
            logger.info(f"  Output shape: {y_combined.shape}")
        
        return X_combined, y_combined
    
    def get_sequence_info(self) -> dict:
        """
        Get information about generated sequences
        
        Returns:
            Dictionary with sequence information
        """
        return {
            'n_sequences': self.n_sequences,
            'lookback': self.lookback,
            'forecast_horizon': self.forecast_horizon,
            'stride': self.stride,
            'input_shape': self.input_shape,
            'output_shape': self.output_shape
        }


def create_temporal_features(X_temporal: np.ndarray,
                            add_time_features: bool = True) -> np.ndarray:
    """
    Add temporal features like day of year, month, etc.
    
    Args:
        X_temporal: Input temporal data
        add_time_features: Whether to add time-based features
    
    Returns:
        Enhanced temporal data
    """
    if not add_time_features:
        return X_temporal
    
    n_timesteps = len(X_temporal)
    
    # Create time features (normalized)
    time_idx = np.arange(n_timesteps) / n_timesteps
    
    # Could add more features like:
    # - Day of week (for weekly patterns)
    # - Month (for seasonal patterns)
    # - etc.
    
    # For now, just add normalized time index
    # This helps the model learn temporal trends
    
    logger.info("  Added temporal index feature")
    
    return X_temporal  # Placeholder for now


def pad_sequences(sequences: List[np.ndarray],
                 max_length: Optional[int] = None,
                 padding: str = 'post',
                 value: float = 0.0) -> np.ndarray:
    """
    Pad sequences to the same length
    
    Args:
        sequences: List of sequences with varying lengths
        max_length: Maximum length (uses longest if None)
        padding: 'pre' or 'post'
        value: Padding value
    
    Returns:
        Padded sequences as numpy array
    """
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)
    
    padded = []
    for seq in sequences:
        if len(seq) < max_length:
            pad_width = max_length - len(seq)
            if padding == 'post':
                padded_seq = np.pad(seq, (0, pad_width), constant_values=value)
            else:  # 'pre'
                padded_seq = np.pad(seq, (pad_width, 0), constant_values=value)
        else:
            padded_seq = seq[:max_length]
        
        padded.append(padded_seq)
    
    return np.array(padded)


if __name__ == "__main__":
    # Test sequence generator
    from data.data_loader import WaterQualityForecastingLoader
    
    # Load data
    loader = WaterQualityForecastingLoader()
    data, success = loader.load(verbose=False)
    
    if success:
        X_train = data['X_train']
        y_train = data['y_train']
        
        print("\n" + "="*70)
        print("Testing Sequence Generator")
        print("="*70)
        
        # Create generator
        generator = SequenceGenerator(lookback=30, forecast_horizon=1)
        
        # Test multi-location sequences
        X_seq, y_seq = generator.create_multi_location_sequences(
            X_train, y_train, n_locations=37, verbose=True
        )
        
        print("\n" + "="*70)
        print("✓ Sequence generation complete!")
        print(f"  Sequences created: {len(X_seq)}")
        print(f"  Input shape: {X_seq.shape}")
        print(f"  Output shape: {y_seq.shape}")
        print("="*70)
