"""
============================================================================
Classification Pipeline
Orchestrates data loading, preprocessing, model training, and evaluation
for the complete water potability classification workflow.
============================================================================
"""

import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

sys.path.append(str(Path(__file__).parent.parent))

from config import METRICS_DIR, RANDOM_SEED, CLASSIFICATION_CONFIG
from utils.logger import get_logger
from utils.file_handler import FileHandler
from utils.helpers import set_random_seeds
from data.data_loader import WaterPotabilityLoader
from data.preprocessing_classification import preprocess_classification_data
from models.classification.random_forest_model import RandomForestModel
from models.classification.svm_model import SVMModel
from models.classification.ensemble_model import EnsembleModel

logger = get_logger()


def run_classification_pipeline(save_models: bool = True,
                                 verbose: bool = True) -> Dict[str, Any]:
    """
    Full classification pipeline:
    1. Load data
    2. Preprocess (impute, scale, SMOTE, split)
    3. Train RF, XGBoost, SVM, Ensemble
    4. Evaluate all models
    5. Save models and metrics

    Returns:
        Complete results dictionary
    """
    set_random_seeds(RANDOM_SEED)
    pipeline_start = time.time()

    if verbose:
        logger.section("CLASSIFICATION PIPELINE")

    # -------------------------------------------------------------------------
    # STEP 1: Load Data
    # -------------------------------------------------------------------------
    if verbose:
        logger.info("\n[1/5] Loading dataset...")

    loader = WaterPotabilityLoader()
    data, success = loader.load(verbose=False)

    if not success:
        logger.error("Failed to load classification dataset!")
        return {'success': False}

    X, y = loader.get_features_and_target()
    logger.success(f"  Loaded {len(X)} samples, {X.shape[1]} features")

    # -------------------------------------------------------------------------
    # STEP 2: Preprocess
    # -------------------------------------------------------------------------
    if verbose:
        logger.info("\n[2/5] Preprocessing data...")

    prep_results = preprocess_classification_data(
        X, y,
        apply_smote=True,
        save_results=save_models,
        verbose=verbose
    )

    X_train = prep_results['X_train']
    X_test = prep_results['X_test']
    y_train = prep_results['y_train']
    y_test = prep_results['y_test']
    feature_names = prep_results['feature_names']
    preprocessor = prep_results['preprocessor']

    logger.success(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    # -------------------------------------------------------------------------
    # STEP 3: Build Models
    # -------------------------------------------------------------------------
    if verbose:
        logger.info("\n[3/5] Building models...")

    rf_model = RandomForestModel()
    svm_model = SVMModel()

    # Try to build XGBoost (optional)
    xgb_model = None
    try:
        from models.classification.xgboost_model import XGBoostModel
        xgb_model = XGBoostModel()
        logger.success("  XGBoost available")
    except ImportError:
        logger.warning("  XGBoost not available (install: pip install xgboost)")

    # -------------------------------------------------------------------------
    # STEP 4: Train all models
    # -------------------------------------------------------------------------
    if verbose:
        logger.info("\n[4/5] Training models...")

    train_results = {}

    # Random Forest
    logger.info("\n  [1/3] Random Forest...")
    t = time.time()
    rf_model.train(X_train, y_train, feature_names=feature_names, verbose=verbose)
    train_results['random_forest'] = {'train_time': time.time() - t}

    # XGBoost (if available)
    if xgb_model:
        logger.info("\n  [2/3] XGBoost...")
        t = time.time()
        xgb_model.train(X_train, y_train, feature_names=feature_names, verbose=verbose)
        train_results['xgboost'] = {'train_time': time.time() - t}

    # SVM
    logger.info("\n  [3/3] SVM...")
    t = time.time()
    svm_model.train(X_train, y_train, feature_names=feature_names, verbose=verbose)
    train_results['svm'] = {'train_time': time.time() - t}

    # -------------------------------------------------------------------------
    # STEP 5: Evaluate all models + Ensemble
    # -------------------------------------------------------------------------
    if verbose:
        logger.info("\n[5/5] Evaluating all models...")

    eval_results = {}

    rf_metrics = rf_model.evaluate(X_test, y_test, verbose=verbose)
    eval_results['random_forest'] = rf_metrics

    if xgb_model:
        xgb_metrics = xgb_model.evaluate(X_test, y_test, verbose=verbose)
        eval_results['xgboost'] = xgb_metrics

    svm_metrics = svm_model.evaluate(X_test, y_test, verbose=verbose)
    eval_results['svm'] = svm_metrics

    # Build ensemble
    ensemble = EnsembleModel(
        rf_model=rf_model,
        xgb_model=xgb_model,
        svm_model=svm_model
    )
    ensemble_metrics = ensemble.evaluate(X_test, y_test, verbose=verbose)
    eval_results['ensemble'] = ensemble_metrics

    # Print comparison table
    ensemble._print_comparison(eval_results)

    # Feature importance
    rf_importance = rf_model.get_feature_importance()
    if verbose:
        logger.subsection("Top Features (Random Forest)")
        for _, row in rf_importance.head(5).iterrows():
            bar = "█" * int(row['importance'] * 100)
            logger.info(f"  {row['feature']:<30} {row['importance']:.4f} {bar}")

    # -------------------------------------------------------------------------
    # Save models and metrics
    # -------------------------------------------------------------------------
    if save_models:
        logger.subsection("Saving Models")
        rf_model.save()
        if xgb_model:
            xgb_model.save()
        svm_model.save()

        # Save serializable metrics (exclude numpy arrays)
        save_metrics = {}
        for model_name, m in eval_results.items():
            save_metrics[model_name] = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in m.items()
                if k not in ('y_pred', 'y_prob', 'confusion_matrix', 'classification_report')
            }
            save_metrics[model_name]['confusion_matrix'] = m.get('confusion_matrix', [])

        FileHandler.save_json(save_metrics, METRICS_DIR / 'classification_metrics.json')
        logger.success("Models and metrics saved!")

    total_time = time.time() - pipeline_start

    logger.section("CLASSIFICATION PIPELINE COMPLETE")
    logger.info(f"  Total time: {total_time:.1f}s")
    logger.info(f"  Best model: {max(eval_results, key=lambda k: eval_results[k].get('f1_score', 0))}")
    logger.info(f"  Best F1: {max(m.get('f1_score', 0) for m in eval_results.values()):.4f}")
    logger.info(f"  Best AUC: {max(m.get('roc_auc', 0) for m in eval_results.values()):.4f}")

    return {
        'success': True,
        'eval_results': eval_results,
        'rf_importance': rf_importance,
        'xgb_importance': xgb_model.get_feature_importance() if xgb_model else None,
        'models': {
            'rf': rf_model,
            'xgb': xgb_model,
            'svm': svm_model,
            'ensemble': ensemble
        },
        'preprocessor': preprocessor,
        'feature_names': feature_names,
        'total_time': total_time
    }


if __name__ == "__main__":
    results = run_classification_pipeline(save_models=True, verbose=True)

    if results['success']:
        print("\n" + "="*70)
        print("✓ Classification pipeline complete!")
        print(f"  Total time: {results['total_time']:.1f}s")
        best = max(results['eval_results'],
                   key=lambda k: results['eval_results'][k].get('f1_score', 0))
        print(f"  Best model: {best}")
        print("="*70)
