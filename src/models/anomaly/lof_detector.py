"""
============================================================================
Local Outlier Factor (LOF) Anomaly Detector
Detects anomalies based on local density deviation
============================================================================
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import precision_score, recall_score, f1_score

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import ANOMALY_CONFIG, MODEL_PATHS, RANDOM_SEED
from utils.logger import get_logger
from utils.file_handler import FileHandler

logger = get_logger()


class LOFDetector:
    """
    Local Outlier Factor for anomaly detection
    
    How it works:
    - Compares local density of a point with its neighbors
    - Points in sparse regions = anomalies
    - Good for detecting local deviations
    
    Best for: Contamination events that create density changes
    """

    def __init__(self, config: Dict = None):
        """Initialize LOF detector"""
        self.config = config or ANOMALY_CONFIG
        self.model = None
        self.metrics = {}
        self._build_model()

    def _build_model(self):
        """Build LOF model"""
        self.model = LocalOutlierFactor(
            n_neighbors=self.config.get('lof_n_neighbors', 20),
            contamination=self.config.get('lof_contamination', 0.05),
            novelty=False,  # For fit_predict mode
            n_jobs=-1
        )
        logger.info("  LOF built:")
        logger.info(f"    Neighbors: {self.config.get('lof_n_neighbors', 20)}")
        logger.info(f"    Contamination: {self.config.get('lof_contamination', 0.05)}")

    def fit_predict(self, X: np.ndarray, verbose: bool = True) -> np.ndarray:
        """
        Fit and predict in one step (LOF requires this)
        
        Args:
            X: Data to analyze
            verbose: Print progress
        
        Returns:
            Predictions: 1=normal, -1=anomaly
        """
        if verbose:
            logger.subsection("Running LOF Detection")

        predictions = self.model.fit_predict(X)
        scores = self.model.negative_outlier_factor_

        n_anomalies = np.sum(predictions == -1)
        anomaly_pct = (n_anomalies / len(X)) * 100

        self.metrics['n_samples'] = len(X)
        self.metrics['n_anomalies'] = n_anomalies
        self.metrics['anomaly_percentage'] = anomaly_pct

        if verbose:
            logger.success(f"  Analyzed {len(X)} samples")
            logger.info(f"  Anomalies: {n_anomalies} ({anomaly_pct:.2f}%)")

        return predictions

    def predict_binary(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies as binary
        
        Returns:
            Binary predictions: 0=normal, 1=anomaly
        """
        predictions = self.fit_predict(X, verbose=False)
        return (predictions == -1).astype(int)

    def get_scores(self) -> np.ndarray:
        """
        Get anomaly scores (more negative = more anomalous)
        
        Returns:
            Negative outlier factors
        """
        if self.model.negative_outlier_factor_ is None:
            raise ValueError("Model not fitted. Call fit_predict() first.")
        return self.model.negative_outlier_factor_

    def evaluate(self, X: np.ndarray, y_true: np.ndarray = None,
                 verbose: bool = True) -> Dict:
        """
        Evaluate LOF performance
        
        Args:
            X: Test data
            y_true: True labels (1=anomaly, 0=normal) - optional
            verbose: Print results
        
        Returns:
            Metrics dictionary
        """
        predictions = self.predict_binary(X)
        scores = self.get_scores()

        metrics = {
            'n_samples': len(X),
            'n_anomalies_detected': int(np.sum(predictions)),
            'anomaly_rate': float(np.mean(predictions)),
            'predictions': predictions,
            'scores': scores
        }

        if y_true is not None:
            metrics['precision'] = precision_score(y_true, predictions, zero_division=0)
            metrics['recall'] = recall_score(y_true, predictions, zero_division=0)
            metrics['f1_score'] = f1_score(y_true, predictions, zero_division=0)

            if verbose:
                logger.subsection("LOF Evaluation")
                logger.info(f"  Precision: {metrics['precision']:.4f}")
                logger.info(f"  Recall:    {metrics['recall']:.4f}")
                logger.info(f"  F1-Score:  {metrics['f1_score']:.4f}")
                logger.info(f"  Detected:  {metrics['n_anomalies_detected']} / {len(X)}")
        else:
            if verbose:
                logger.subsection("LOF Detection")
                logger.info(f"  Anomalies: {metrics['n_anomalies_detected']} ({metrics['anomaly_rate']*100:.2f}%)")

        self.metrics.update(metrics)
        return metrics

    def save(self, filepath: Path = None):
        """Save model (LOF state is limited)"""
        filepath = filepath or MODEL_PATHS['lof']
        # Save the fitted model
        FileHandler.save_model_joblib(self.model, filepath)

    def load(self, filepath: Path = None):
        """Load model"""
        filepath = filepath or MODEL_PATHS['lof']
        self.model = FileHandler.load_model_joblib(filepath)
        return self


if __name__ == "__main__":
    # Test
    np.random.seed(42)
    X_normal = np.random.randn(1000, 10)
    X_anomaly = np.random.randn(50, 10) + 4
    X_test = np.vstack([X_normal[:100], X_anomaly[:10]])
    y_test = np.array([0]*100 + [1]*10)

    detector = LOFDetector()
    metrics = detector.evaluate(X_test, y_test)

    print(f"\n✓ LOF test passed!")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
