"""
============================================================================
Anomaly Detection Pipeline
Orchestrates all anomaly detectors on water quality data
============================================================================
"""

import sys
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any

sys.path.append(str(Path(__file__).parent.parent))

from config import METRICS_DIR, RANDOM_SEED
from utils.logger import get_logger
from utils.file_handler import FileHandler
from utils.helpers import set_random_seeds
from data.data_loader import WaterPotabilityLoader
from data.preprocessing_classification import ClassificationPreprocessor
from models.anomaly.ensemble_anomaly import EnsembleAnomalyDetector

logger = get_logger()


def run_anomaly_pipeline(save_models: bool = True,
                         use_autoencoder: bool = True,
                         verbose: bool = True) -> Dict[str, Any]:
    """
    Full anomaly detection pipeline:
    1. Load and preprocess water quality data
    2. Train all 4 detectors (IF, LOF, OCSVM, AE)
    3. Evaluate ensemble
    4. Save models and results
    
    Returns:
        Complete results dictionary
    """
    set_random_seeds(RANDOM_SEED)
    pipeline_start = time.time()

    if verbose:
        logger.section("ANOMALY DETECTION PIPELINE")

    # -------------------------------------------------------------------------
    # STEP 1: Load Data
    # -------------------------------------------------------------------------
    if verbose:
        logger.info("\n[1/4] Loading water quality data...")

    loader = WaterPotabilityLoader()
    data, success = loader.load(verbose=False)

    if not success:
        logger.error("Failed to load data!")
        return {'success': False}

    X, y = loader.get_features_and_target()
    logger.success(f"  Loaded {len(X)} samples")

    # -------------------------------------------------------------------------
    # STEP 2: Preprocess
    # -------------------------------------------------------------------------
    if verbose:
        logger.info("\n[2/4] Preprocessing...")

    preprocessor = ClassificationPreprocessor()
    X_processed, _ = preprocessor.fit_transform(X, y, verbose=False)

    logger.success(f"  Preprocessed: {X_processed.shape}")

    # -------------------------------------------------------------------------
    # STEP 3: Train Ensemble
    # -------------------------------------------------------------------------
    if verbose:
        logger.info("\n[3/4] Training anomaly detectors...")

    ensemble = EnsembleAnomalyDetector(use_autoencoder=use_autoencoder)

    # Use only clean data for training (potable water = normal)
    # We'll simulate by taking a subset
    # In real deployment, you'd use verified normal data
    X_train = X_processed.values if hasattr(X_processed, 'values') else X_processed

    # Train on all data (unsupervised learning)
    ensemble.fit(X_train, verbose=verbose)

    # -------------------------------------------------------------------------
    # STEP 4: Evaluate & Detect
    # -------------------------------------------------------------------------
    if verbose:
        logger.info("\n[4/4] Running anomaly detection...")

    # Run detection on the data
    # In real scenario, you'd have separate test/deployment data
    metrics = ensemble.evaluate(X_train, verbose=verbose)

    # Get detailed vote analysis
    vote_counts = ensemble.get_vote_counts(X_train)
    predictions = ensemble.predict_binary(X_train)

    # Analyze vote distribution
    vote_distribution = {}
    for i in range(len(ensemble.detectors) + 1):
        count = np.sum(vote_counts == i)
        vote_distribution[f'{i}_votes'] = int(count)

    if verbose:
        logger.subsection("Vote Distribution")
        for i in range(len(ensemble.detectors) + 1):
            count = vote_distribution[f'{i}_votes']
            pct = (count / len(X_train)) * 100
            logger.info(f"  {i} detectors flagged: {count:>5} ({pct:>5.2f}%)")

    # Save results
    if save_models:
        logger.subsection("Saving Models & Results")

        # Save models
        try:
            ensemble.save_all()
        except Exception as e:
            logger.warning(f"  Some models couldn't be saved: {e}")

        # Save metrics
        save_metrics = {
            'ensemble': {
                'n_samples': metrics['n_samples'],
                'n_anomalies_detected': metrics['n_anomalies_detected'],
                'anomaly_rate': metrics['anomaly_rate'],
                'voting_threshold': ensemble.voting_threshold
            },
            'individual': metrics['individual'],
            'vote_distribution': vote_distribution
        }

        FileHandler.save_json(save_metrics, METRICS_DIR / 'anomaly_metrics.json')
        logger.success("Results saved!")

    total_time = time.time() - pipeline_start

    logger.section("ANOMALY DETECTION PIPELINE COMPLETE")
    logger.info(f"  Total time: {total_time:.1f}s")
    logger.info(f"  Samples analyzed: {len(X_train)}")
    logger.info(f"  Anomalies found: {metrics['n_anomalies_detected']} ({metrics['anomaly_rate']*100:.2f}%)")

    return {
        'success': True,
        'ensemble': ensemble,
        'metrics': metrics,
        'vote_counts': vote_counts,
        'predictions': predictions,
        'vote_distribution': vote_distribution,
        'total_time': total_time
    }


if __name__ == "__main__":
    results = run_anomaly_pipeline(
        save_models=True,
        use_autoencoder=True,  # Set False if TensorFlow not installed
        verbose=True
    )

    if results['success']:
        print("\n" + "="*70)
        print("✓ Anomaly detection pipeline complete!")
        print(f"  Anomalies detected: {results['metrics']['n_anomalies_detected']}")
        print(f"  Total time: {results['total_time']:.1f}s")
        print("="*70)
