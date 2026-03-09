"""
============================================================================
Ensemble Anomaly Detector
Combines Isolation Forest + LOF + One-Class SVM + Autoencoder
Uses voting for robust anomaly detection
============================================================================
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import ANOMALY_CONFIG, RANDOM_SEED
from utils.logger import get_logger
from models.anomaly.isolation_forest import IsolationForestDetector
from models.anomaly.lof_detector import LOFDetector
from models.anomaly.ocsvm_detector import OneClassSVMDetector

logger = get_logger()


class EnsembleAnomalyDetector:
    """
    Ensemble of multiple anomaly detectors
    
    Strategy:
    - Run all detectors independently
    - Combine predictions via voting
    - Threshold: if ≥N detectors flag it → anomaly
    
    Advantages:
    - More robust than single detector
    - Reduces false positives
    - Catches different types of anomalies
    """

    def __init__(self, config: Dict = None,
                 use_autoencoder: bool = True):
        """
        Initialize ensemble
        
        Args:
            config: Configuration
            use_autoencoder: Whether to include autoencoder (requires TensorFlow)
        """
        self.config = config or ANOMALY_CONFIG
        self.use_autoencoder = use_autoencoder

        # Initialize detectors
        self.if_detector = IsolationForestDetector(config)
        self.lof_detector = LOFDetector(config)
        self.ocsvm_detector = OneClassSVMDetector(config)
        self.ae_detector = None

        if use_autoencoder:
            try:
                from models.anomaly.autoencoder import AutoencoderDetector
                self.ae_detector = AutoencoderDetector(config)
            except ImportError:
                logger.warning("  Autoencoder skipped (TensorFlow not installed)")
                self.use_autoencoder = False

        self.detectors = [
            ('Isolation Forest', self.if_detector),
            ('LOF', self.lof_detector),
            ('One-Class SVM', self.ocsvm_detector),
        ]
        if self.ae_detector:
            self.detectors.append(('Autoencoder', self.ae_detector))

        self.voting_threshold = self.config.get('ensemble_voting_threshold', 0.5)
        self.metrics = {}

    def fit(self, X: np.ndarray, verbose: bool = True) -> Dict:
        """
        Train all detectors
        
        Args:
            X: Normal training data
            verbose: Print progress
        
        Returns:
            Training stats
        """
        if verbose:
            logger.section(f"TRAINING ENSEMBLE ({len(self.detectors)} Detectors)")

        # Train each detector
        for i, (name, detector) in enumerate(self.detectors, 1):
            if verbose:
                logger.info(f"\n[{i}/{len(self.detectors)}] {name}...")

            if name == 'LOF':
                # LOF uses fit_predict
                detector.fit_predict(X, verbose=verbose)
            elif name == 'Autoencoder':
                # Split for validation
                split = int(len(X) * 0.8)
                detector.fit(X[:split], X[split:], verbose=verbose)
            else:
                detector.fit(X, verbose=verbose)

        if verbose:
            logger.success("All detectors trained!")

        return self.metrics

    def predict_binary(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using ensemble voting
        
        Returns:
            Binary predictions: 0=normal, 1=anomaly
        """
        all_predictions = []

        for name, detector in self.detectors:
            if name == 'LOF':
                preds = detector.predict_binary(X)
            else:
                preds = detector.predict_binary(X)
            all_predictions.append(preds)

        # Stack predictions: (n_samples, n_detectors)
        votes = np.column_stack(all_predictions)

        # Voting: percentage of detectors that flagged as anomaly
        vote_ratios = np.mean(votes, axis=1)

        # Threshold
        ensemble_predictions = (vote_ratios >= self.voting_threshold).astype(int)

        return ensemble_predictions

    def get_vote_counts(self, X: np.ndarray) -> np.ndarray:
        """
        Get number of detectors that flagged each sample
        
        Returns:
            Vote counts for each sample
        """
        all_predictions = []
        for name, detector in self.detectors:
            preds = detector.predict_binary(X)
            all_predictions.append(preds)

        votes = np.column_stack(all_predictions)
        return np.sum(votes, axis=1)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray = None,
                 verbose: bool = True) -> Dict:
        """
        Evaluate ensemble and individual detectors
        
        Returns:
            Comprehensive metrics
        """
        from sklearn.metrics import precision_score, recall_score, f1_score

        # Ensemble predictions
        ensemble_preds = self.predict_binary(X_test)
        vote_counts = self.get_vote_counts(X_test)

        metrics = {
            'n_samples': len(X_test),
            'n_anomalies_detected': int(np.sum(ensemble_preds)),
            'anomaly_rate': float(np.mean(ensemble_preds)),
            'predictions': ensemble_preds,
            'vote_counts': vote_counts,
            'individual': {}
        }

        # Evaluate individual detectors
        for name, detector in self.detectors:
            if name == 'LOF':
                preds = detector.predict_binary(X_test)
            else:
                preds = detector.predict_binary(X_test)

            ind_metrics = {
                'n_anomalies': int(np.sum(preds)),
                'anomaly_rate': float(np.mean(preds))
            }

            if y_test is not None:
                ind_metrics['precision'] = precision_score(y_test, preds, zero_division=0)
                ind_metrics['recall'] = recall_score(y_test, preds, zero_division=0)
                ind_metrics['f1_score'] = f1_score(y_test, preds, zero_division=0)

            metrics['individual'][name] = ind_metrics

        # Ensemble metrics
        if y_test is not None:
            metrics['precision'] = precision_score(y_test, ensemble_preds, zero_division=0)
            metrics['recall'] = recall_score(y_test, ensemble_preds, zero_division=0)
            metrics['f1_score'] = f1_score(y_test, ensemble_preds, zero_division=0)

            if verbose:
                self._print_comparison(metrics, y_test)
        else:
            if verbose:
                logger.subsection("Ensemble Detection Results")
                logger.info(f"  Anomalies: {metrics['n_anomalies_detected']} ({metrics['anomaly_rate']*100:.2f}%)")
                logger.info(f"  Voting threshold: {self.voting_threshold}")

        self.metrics.update(metrics)
        return metrics

    def _print_comparison(self, metrics: Dict, y_test: np.ndarray):
        """Print comparison table"""
        logger.section("ANOMALY DETECTION COMPARISON")

        header = f"  {'Detector':<20} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Detected':>10}"
        logger.info(header)
        logger.info("  " + "-" * 70)

        # Individual detectors
        for name in ['Isolation Forest', 'LOF', 'One-Class SVM', 'Autoencoder']:
            if name not in metrics['individual']:
                continue
            m = metrics['individual'][name]
            if 'precision' in m:
                logger.info(
                    f"  {name:<20} "
                    f"{m['precision']:>10.4f} "
                    f"{m['recall']:>10.4f} "
                    f"{m['f1_score']:>10.4f} "
                    f"{m['n_anomalies']:>10}"
                )

        # Ensemble
        logger.info("  " + "-" * 70)
        logger.info(
            f"  {'Ensemble (Voting)':<20} "
            f"{metrics['precision']:>10.4f} "
            f"{metrics['recall']:>10.4f} "
            f"{metrics['f1_score']:>10.4f} "
            f"{metrics['n_anomalies_detected']:>10}"
        )

        # Best detector
        best = max(metrics['individual'].items(),
                  key=lambda x: x[1].get('f1_score', 0))
        logger.info(f"\n  Best individual: {best[0]} (F1={best[1].get('f1_score', 0):.4f})")

    def save_all(self):
        """Save all detectors"""
        logger.subsection("Saving Anomaly Detectors")
        for name, detector in self.detectors:
            if name != 'LOF':  # LOF save is tricky
                detector.save()
        logger.success("All detectors saved!")

    def load_all(self):
        """Load all detectors"""
        logger.subsection("Loading Anomaly Detectors")
        for name, detector in self.detectors:
            if name != 'LOF':
                detector.load()
        logger.success("All detectors loaded!")


if __name__ == "__main__":
    # Test
    np.random.seed(42)
    X_normal = np.random.randn(1000, 10).astype(np.float32)
    X_anomaly = (np.random.randn(50, 10) + 3).astype(np.float32)
    X_test = np.vstack([X_normal[:100], X_anomaly[:10]])
    y_test = np.array([0]*100 + [1]*10)

    ensemble = EnsembleAnomalyDetector(use_autoencoder=False)  # Skip AE for quick test
    ensemble.fit(X_normal)
    metrics = ensemble.evaluate(X_test, y_test)

    print(f"\n✓ Ensemble test passed!")
    print(f"  Ensemble F1: {metrics['f1_score']:.4f}")
