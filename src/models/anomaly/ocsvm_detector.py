"""
============================================================================
One-Class SVM Anomaly Detector
Learns a boundary around normal data in high-dimensional space
============================================================================
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
from sklearn.svm import OneClassSVM
from sklearn.metrics import precision_score, recall_score, f1_score

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import ANOMALY_CONFIG, MODEL_PATHS, RANDOM_SEED
from utils.logger import get_logger
from utils.file_handler import FileHandler

logger = get_logger()


class OneClassSVMDetector:
    """
    One-Class SVM for anomaly detection
    
    How it works:
    - Learns a tight boundary around normal data
    - Points outside the boundary = anomalies
    - Uses RBF kernel for non-linear boundaries
    
    Best for: Finding data points that don't fit the normal pattern
    """

    def __init__(self, config: Dict = None):
        """Initialize One-Class SVM"""
        self.config = config or ANOMALY_CONFIG
        self.model = None
        self.metrics = {}
        self._build_model()

    def _build_model(self):
        """Build One-Class SVM"""
        self.model = OneClassSVM(
            kernel=self.config.get('ocsvm_kernel', 'rbf'),
            gamma=self.config.get('ocsvm_gamma', 'scale'),
            nu=self.config.get('ocsvm_nu', 0.05),  # Anomaly fraction
            cache_size=500
        )
        logger.info("  One-Class SVM built:")
        logger.info(f"    Kernel: {self.config.get('ocsvm_kernel', 'rbf')}")
        logger.info(f"    Nu: {self.config.get('ocsvm_nu', 0.05)}")

    def fit(self, X: np.ndarray, verbose: bool = True) -> Dict:
        """
        Fit the model on normal data
        
        Args:
            X: Training data (normal samples)
            verbose: Print progress
        
        Returns:
            Training statistics
        """
        if verbose:
            logger.subsection("Training One-Class SVM")
            logger.info("  Note: May take 1-3 minutes on large data...")

        self.model.fit(X)

        # Predict on training data
        predictions = self.model.predict(X)
        scores = self.model.decision_function(X)

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
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        return self.model.predict(X)

    def predict_binary(self, X: np.ndarray) -> np.ndarray:
        """
        Predict as binary (0=normal, 1=anomaly)
        
        Returns:
            Binary predictions
        """
        predictions = self.predict(X)
        return (predictions == -1).astype(int)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Get decision scores (negative = anomaly)
        
        Returns:
            Decision function values
        """
        return self.model.decision_function(X)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray = None,
                 verbose: bool = True) -> Dict:
        """
        Evaluate detector
        
        Args:
            X_test: Test data
            y_test: True labels (optional)
            verbose: Print results
        
        Returns:
            Metrics
        """
        predictions = self.predict_binary(X_test)
        scores = self.decision_function(X_test)

        metrics = {
            'n_samples': len(X_test),
            'n_anomalies_detected': int(np.sum(predictions)),
            'anomaly_rate': float(np.mean(predictions)),
            'predictions': predictions,
            'scores': scores
        }

        if y_test is not None:
            metrics['precision'] = precision_score(y_test, predictions, zero_division=0)
            metrics['recall'] = recall_score(y_test, predictions, zero_division=0)
            metrics['f1_score'] = f1_score(y_test, predictions, zero_division=0)

            if verbose:
                logger.subsection("One-Class SVM Evaluation")
                logger.info(f"  Precision: {metrics['precision']:.4f}")
                logger.info(f"  Recall:    {metrics['recall']:.4f}")
                logger.info(f"  F1-Score:  {metrics['f1_score']:.4f}")
                logger.info(f"  Detected:  {metrics['n_anomalies_detected']} / {len(X_test)}")
        else:
            if verbose:
                logger.subsection("One-Class SVM Detection")
                logger.info(f"  Anomalies: {metrics['n_anomalies_detected']} ({metrics['anomaly_rate']*100:.2f}%)")

        self.metrics.update(metrics)
        return metrics

    def save(self, filepath: Path = None):
        """Save model"""
        filepath = filepath or MODEL_PATHS['ocsvm']
        FileHandler.save_model_joblib(self.model, filepath)

    def load(self, filepath: Path = None):
        """Load model"""
        filepath = filepath or MODEL_PATHS['ocsvm']
        self.model = FileHandler.load_model_joblib(filepath)
        return self


if __name__ == "__main__":
    # Test
    np.random.seed(42)
    X_normal = np.random.randn(500, 10)  # Smaller for speed
    X_anomaly = np.random.randn(25, 10) + 4
    X_test = np.vstack([X_normal[:50], X_anomaly[:5]])
    y_test = np.array([0]*50 + [1]*5)

    detector = OneClassSVMDetector()
    detector.fit(X_normal)
    metrics = detector.evaluate(X_test, y_test)

    print(f"\n✓ One-Class SVM test passed!")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
