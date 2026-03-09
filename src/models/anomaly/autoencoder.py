"""
============================================================================
Autoencoder Anomaly Detector
Deep learning approach - reconstructs normal data, fails on anomalies
============================================================================
"""

import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import ANOMALY_CONFIG, MODEL_PATHS, RANDOM_SEED
from utils.logger import get_logger
from utils.file_handler import FileHandler

logger = get_logger()


def build_autoencoder(input_dim: int, encoding_dim: int = 8) -> 'Model':
    """
    Build autoencoder architecture
    
    Architecture:
      Input → Dense(16) → Dense(encoding_dim) → Dense(16) → Dense(input_dim)
                  Encoder                            Decoder
    
    Args:
        input_dim: Number of features
        encoding_dim: Compressed representation size
    
    Returns:
        Keras model
    """
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        tf.random.set_seed(RANDOM_SEED)
    except ImportError:
        raise ImportError("TensorFlow required. Install: pip install tensorflow")

    inputs = keras.Input(shape=(input_dim,))

    # Encoder
    x = layers.Dense(16, activation='relu')(inputs)
    encoded = layers.Dense(encoding_dim, activation='relu', name='encoding')(x)

    # Decoder
    x = layers.Dense(16, activation='relu')(encoded)
    outputs = layers.Dense(input_dim, activation='linear')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='Autoencoder')
    model.compile(optimizer='adam', loss='mse')

    return model


class AutoencoderDetector:
    """
    Autoencoder for anomaly detection
    
    How it works:
    - Learns to reconstruct normal data
    - High reconstruction error = anomaly
    - Deep learning captures complex patterns
    
    Best for: Complex, subtle anomalies
    """

    def __init__(self, config: Dict = None):
        """Initialize Autoencoder"""
        self.config = config or ANOMALY_CONFIG
        self.model = None
        self.threshold = None
        self.metrics = {}
        self.input_dim = None

    def build(self, input_dim: int, verbose: bool = True):
        """Build autoencoder"""
        self.input_dim = input_dim
        encoding_dim = self.config.get('ae_encoding_dim', 8)

        self.model = build_autoencoder(input_dim, encoding_dim)

        if verbose:
            logger.info("  Autoencoder built:")
            logger.info(f"    Input: {input_dim} → Encoding: {encoding_dim} → Output: {input_dim}")
            logger.info(f"    Parameters: {self.model.count_params():,}")

    def fit(self, X: np.ndarray, X_val: np.ndarray = None,
            verbose: bool = True) -> Dict:
        """
        Train autoencoder on normal data
        
        Args:
            X: Training data (normal samples)
            X_val: Validation data (optional)
            verbose: Print progress
        
        Returns:
            Training history
        """
        if self.model is None:
            self.build(X.shape[1], verbose=verbose)

        if verbose:
            logger.subsection("Training Autoencoder")

        from tensorflow.keras.callbacks import EarlyStopping

        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=self.config.get('ae_patience', 10),
                restore_best_weights=True,
                verbose=0
            )
        ]

        validation_data = (X_val, X_val) if X_val is not None else None

        history = self.model.fit(
            X, X,  # Autoencoder: input = output
            epochs=self.config.get('ae_epochs', 100),
            batch_size=self.config.get('ae_batch_size', 32),
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1 if verbose else 0
        )

        # Set threshold based on training reconstruction error
        train_errors = self._compute_errors(X)
        self.threshold = np.percentile(train_errors,
                                      self.config.get('ae_threshold_percentile', 95))

        if verbose:
            logger.success(f"  Training complete")
            logger.info(f"  Threshold: {self.threshold:.6f} (95th percentile)")

        self.metrics['threshold'] = float(self.threshold)
        self.metrics['train_loss'] = float(history.history['loss'][-1])
        return self.metrics

    def _compute_errors(self, X: np.ndarray) -> np.ndarray:
        """Compute reconstruction errors"""
        reconstructions = self.model.predict(X, verbose=0)
        errors = np.mean(np.square(X - reconstructions), axis=1)
        return errors

    def predict_binary(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies (0=normal, 1=anomaly)
        
        Returns:
            Binary predictions
        """
        if self.threshold is None:
            raise ValueError("Model not trained. Call fit() first.")

        errors = self._compute_errors(X)
        return (errors > self.threshold).astype(int)

    def get_scores(self, X: np.ndarray) -> np.ndarray:
        """Get reconstruction errors (anomaly scores)"""
        return self._compute_errors(X)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray = None,
                 verbose: bool = True) -> Dict:
        """Evaluate detector"""
        from sklearn.metrics import precision_score, recall_score, f1_score

        predictions = self.predict_binary(X_test)
        errors = self.get_scores(X_test)

        metrics = {
            'n_samples': len(X_test),
            'n_anomalies_detected': int(np.sum(predictions)),
            'anomaly_rate': float(np.mean(predictions)),
            'mean_error': float(np.mean(errors)),
            'predictions': predictions,
            'scores': errors
        }

        if y_test is not None:
            metrics['precision'] = precision_score(y_test, predictions, zero_division=0)
            metrics['recall'] = recall_score(y_test, predictions, zero_division=0)
            metrics['f1_score'] = f1_score(y_test, predictions, zero_division=0)

            if verbose:
                logger.subsection("Autoencoder Evaluation")
                logger.info(f"  Precision: {metrics['precision']:.4f}")
                logger.info(f"  Recall:    {metrics['recall']:.4f}")
                logger.info(f"  F1-Score:  {metrics['f1_score']:.4f}")
                logger.info(f"  Detected:  {metrics['n_anomalies_detected']} / {len(X_test)}")
        else:
            if verbose:
                logger.subsection("Autoencoder Detection")
                logger.info(f"  Anomalies: {metrics['n_anomalies_detected']} ({metrics['anomaly_rate']*100:.2f}%)")

        self.metrics.update(metrics)
        return metrics

    def save(self, filepath: Path = None):
        """Save model"""
        filepath = filepath or MODEL_PATHS['autoencoder']
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(filepath))
        # Save threshold separately
        import json
        threshold_file = Path(str(filepath).replace('.h5', '_threshold.json'))
        with open(threshold_file, 'w') as f:
            json.dump({'threshold': float(self.threshold)}, f)
        logger.success(f"Autoencoder saved: {Path(filepath).name}")

    def load(self, filepath: Path = None):
        """Load model"""
        from tensorflow import keras
        filepath = filepath or MODEL_PATHS['autoencoder']
        self.model = keras.models.load_model(str(filepath))
        # Load threshold
        import json
        threshold_file = Path(str(filepath).replace('.h5', '_threshold.json'))
        with open(threshold_file, 'r') as f:
            data = json.load(f)
            self.threshold = data['threshold']
        logger.success(f"Autoencoder loaded: {Path(filepath).name}")
        return self


if __name__ == "__main__":
    # Test
    np.random.seed(42)
    X_normal = np.random.randn(1000, 10).astype(np.float32)
    X_anomaly = (np.random.randn(50, 10) + 3).astype(np.float32)
    X_test = np.vstack([X_normal[:100], X_anomaly[:10]])
    y_test = np.array([0]*100 + [1]*10)

    detector = AutoencoderDetector()
    detector.fit(X_normal[:800], X_normal[800:])
    metrics = detector.evaluate(X_test, y_test)

    print(f"\n✓ Autoencoder test passed!")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
