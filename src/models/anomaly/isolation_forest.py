"""
============================================================================
Isolation Forest Anomaly Detector
Detects anomalous water quality patterns using tree-based isolation
============================================================================
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import ANOMALY_CONFIG, MODEL_PATHS, RANDOM_SEED
from utils.logger import get_logger
from utils.file_handler import FileHandler

logger = get_logger()


class IsolationForestDetector:
    """
    Isolation Forest for anomaly detection
    
    How it works:
    - Randomly partitions data space using trees
    - Anomalies are easier to isolate (shorter paths)
    - Fast, scalable, works well for high-dimensional data
    
    Best for: Detecting rare contamination events
    """

    def __init__(self, config: Dict = None):
        """
        Initialize Isolation Forest detector
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or ANOMALY_CONFIG
        self.model = None
        self.threshold = None
        self.metrics = {}
        self._build_model()

    def _build_model(self):
        """Build Isolation Forest model"""
        self.model = IsolationForest(
            n_estimators=self.config.get('if_n_estimators', 100),
            max_samples=self.config.get('if_max_samples', 'auto'),
            contamination=self.config.get('if_contamination', 0.05),
            max_features=self.config.get('if_max_features', 1.0),
            random_state=RANDOM_SEED,
            n_jobs=-1
        )
        logger.info("  Isolation Forest built:")
        logger.info(f"    Trees: {self.config.get('if_n_estimators', 100)}")
        logger.info(f"    Contamination: {self.config.get('if_contamination', 0.05)}")

    def fit(self, X: np.ndarray, verbose: bool = True) -> Dict:
        """
        Fit the model on normal data
        
        Args:
            X: Training data (should be mostly normal samples)
            verbose: Print progress
        
        Returns:
            Training statistics
        """
        if verbose:
            logger.subsection("Training Isolation Forest")

        self.model.fit(X)

        # Get anomaly scores
        scores = self.model.decision_function(X)
        predictions = self.model.predict(X)

        n_anomalies = np.sum(predictions == -1)
        anomaly_pct = (n_anomalies / len(X)) * 100

        self.metrics['n_train_samples'] = len(X)
        self.metrics['n_anomalies_detected'] = n_anomalies
        self.metrics['anomaly_percentage'] = anomaly_pct

        if verbose:
            logger.success(f"  Trained on {len(X)} samples")
            logger.info(f"  Anomalies detected: {n_anomalies} ({anomaly_pct:.2f}%)")

        return self.metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies (1=normal, -1=anomaly)
        
        Returns:
            Predictions: 1 for normal, -1 for anomaly
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        return self.model.predict(X)

    def predict_binary(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies as binary (0=normal, 1=anomaly)
        
        Returns:
            Binary predictions: 0 for normal, 1 for anomaly
        """
        predictions = self.predict(X)
        return (predictions == -1).astype(int)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Get anomaly scores (lower = more anomalous)
        
        Returns:
            Anomaly scores
        """
        return self.model.decision_function(X)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray = None,
                 verbose: bool = True) -> Dict:
        """
        Evaluate detector performance
        
        Args:
            X_test: Test samples
            y_test: True labels (1=anomaly, 0=normal) - optional
            verbose: Print results
        
        Returns:
            Evaluation metrics
        """
        predictions = self.predict_binary(X_test)
        scores = self.decision_function(X_test)

        metrics = {
            'n_samples': len(X_test),
            'n_anomalies_detected': int(np.sum(predictions)),
            'anomaly_rate': float(np.mean(predictions)),
            'mean_score': float(np.mean(scores)),
            'predictions': predictions,
            'scores': scores
        }

        # If ground truth labels provided
        if y_test is not None:
            metrics['precision'] = precision_score(y_test, predictions, zero_division=0)
            metrics['recall'] = recall_score(y_test, predictions, zero_division=0)
            metrics['f1_score'] = f1_score(y_test, predictions, zero_division=0)

            if verbose:
                logger.subsection("Isolation Forest Evaluation")
                logger.info(f"  Precision: {metrics['precision']:.4f}")
                logger.info(f"  Recall:    {metrics['recall']:.4f}")
                logger.info(f"  F1-Score:  {metrics['f1_score']:.4f}")
                logger.info(f"  Detected:  {metrics['n_anomalies_detected']} / {len(X_test)}")
        else:
            if verbose:
                logger.subsection("Isolation Forest Detection")
                logger.info(f"  Anomalies: {metrics['n_anomalies_detected']} ({metrics['anomaly_rate']*100:.2f}%)")

        self.metrics.update(metrics)
        return metrics

    def save(self, filepath: Path = None):
        """Save model"""
        filepath = filepath or MODEL_PATHS['isolation_forest']
        FileHandler.save_model_joblib(self.model, filepath)

    def load(self, filepath: Path = None):
        """Load model"""
        filepath = filepath or MODEL_PATHS['isolation_forest']
        self.model = FileHandler.load_model_joblib(filepath)
        return self


if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)
    X_normal = np.random.randn(1000, 10)
    X_anomaly = np.random.randn(50, 10) + 5  # Shifted distribution
    X_test = np.vstack([X_normal[:100], X_anomaly[:10]])
    y_test = np.array([0]*100 + [1]*10)

    detector = IsolationForestDetector()
    detector.fit(X_normal)
    metrics = detector.evaluate(X_test, y_test)

    print(f"\n✓ Isolation Forest test passed!")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
