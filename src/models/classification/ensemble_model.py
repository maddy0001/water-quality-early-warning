"""
============================================================================
Ensemble Classification Model
Soft Voting Ensemble: Random Forest + XGBoost + SVM
============================================================================
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report, confusion_matrix
)

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import CLASSIFICATION_CONFIG, MODEL_PATHS, RANDOM_SEED
from utils.logger import get_logger
from utils.file_handler import FileHandler
from models.classification.random_forest_model import RandomForestModel
from models.classification.xgboost_model import XGBoostModel
from models.classification.svm_model import SVMModel

logger = get_logger()


class EnsembleModel:
    """
    Soft Voting Ensemble combining RF + XGBoost + SVM
    
    Strategy:
    - Each model outputs probability
    - Weighted average of probabilities
    - Final prediction = argmax(weighted_probs)
    
    Novelty for Paper:
    - Multi-model fusion for robustness
    - Handles individual model weaknesses
    - State-of-art performance benchmark
    """

    def __init__(self, config: Dict = None,
                 rf_model: RandomForestModel = None,
                 xgb_model: XGBoostModel = None,
                 svm_model: SVMModel = None):
        """
        Initialize ensemble

        Args:
            config: Configuration dict
            rf_model: Pre-trained Random Forest (optional)
            xgb_model: Pre-trained XGBoost (optional)
            svm_model: Pre-trained SVM (optional)
        """
        self.config = config or CLASSIFICATION_CONFIG
        self.rf = rf_model or RandomForestModel(config)
        self.svm = svm_model or SVMModel(config)
        self.metrics = {}
        self.individual_metrics = {}
        self.feature_names = None

        # XGBoost is optional
        if xgb_model is not None:
            self.xgb = xgb_model
        else:
            try:
                self.xgb = XGBoostModel(config)
            except ImportError:
                self.xgb = None
                logger.warning("  XGBoost not available - ensemble will use RF + SVM only")

        # Adjust weights based on available models
        if self.xgb is None:
            self.weights = [0.55, 0.45]   # RF, SVM (no XGB)
            self.model_list = [self.rf, self.svm]
            self.model_names_list = ['random_forest', 'svm']
        else:
            self.weights = self.config.get('ensemble_weights', [0.35, 0.40, 0.25])
            self.model_list = [self.rf, self.xgb, self.svm]
            self.model_names_list = ['random_forest', 'xgboost', 'svm']

        self.voting = self.config.get('ensemble_voting', 'soft')

    def train_all(self, X_train: pd.DataFrame, y_train: pd.Series,
                  feature_names: list = None, verbose: bool = True) -> Dict:
        """Train all available base models"""
        if verbose:
            logger.section(f"TRAINING ENSEMBLE ({len(self.model_list)} Models)")

        self.feature_names = feature_names

        for i, (model, name) in enumerate(zip(self.model_list, self.model_names_list), 1):
            if verbose:
                logger.info(f"\n[{i}/{len(self.model_list)}] {name}...")
            metrics = model.train(X_train, y_train,
                                  feature_names=feature_names, verbose=verbose)
            self.individual_metrics[name] = metrics

        if verbose:
            logger.success("All base models trained!")

        return self.individual_metrics

    def evaluate_all(self, X_test: pd.DataFrame, y_test: pd.Series,
                     verbose: bool = True) -> Dict:
        """Evaluate all individual models and the ensemble"""
        if verbose:
            logger.section("EVALUATING ALL MODELS")

        all_metrics = {}
        for model, name in zip(self.model_list, self.model_names_list):
            m = model.evaluate(X_test, y_test, verbose=verbose)
            all_metrics[name] = m
            self.individual_metrics[name] = m

        # Ensemble
        ensemble_metrics = self.evaluate(X_test, y_test, verbose=verbose)
        all_metrics['ensemble'] = ensemble_metrics

        if verbose:
            self._print_comparison(all_metrics)

        return all_metrics

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series,
                 verbose: bool = True) -> Dict:
        """
        Evaluate the ensemble using soft voting
        
        Returns:
            Ensemble metrics
        """
        y_pred, y_prob = self._predict_ensemble(X_test)

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_prob),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred),
            'y_pred': y_pred,
            'y_prob': y_prob
        }

        self.metrics = metrics

        if verbose:
            logger.subsection("Ensemble Evaluation")
            logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
            logger.info(f"  Precision: {metrics['precision']:.4f}")
            logger.info(f"  Recall:    {metrics['recall']:.4f}")
            logger.info(f"  F1-Score:  {metrics['f1_score']:.4f}")
            logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
            weight_str = ", ".join(
                f"{n}={w}" for n, w in zip(self.model_names_list, self.weights)
            )
            logger.info(f"  Weights: {weight_str}")

        return metrics

    def _predict_ensemble(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Generate ensemble predictions via weighted soft voting"""
        probs = [m.predict_proba(X)[:, 1] for m in self.model_list]
        y_prob = sum(w * p for w, p in zip(self.weights, probs))
        y_pred = (y_prob >= 0.5).astype(int)
        return y_pred, y_prob

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels using ensemble"""
        y_pred, _ = self._predict_ensemble(X)
        return y_pred

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities using ensemble"""
        _, y_prob = self._predict_ensemble(X)
        # Return as 2D array: (n_samples, 2)
        return np.column_stack([1 - y_prob, y_prob])

    def _print_comparison(self, all_metrics: Dict):
        """Print model comparison table"""
        logger.section("MODEL COMPARISON")

        model_names = {
            'random_forest': 'Random Forest',
            'xgboost': 'XGBoost',
            'svm': 'SVM',
            'ensemble': 'Ensemble (Voting)'
        }

        header = f"{'Model':<22} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10}"
        logger.info(header)
        logger.info("-" * 72)

        for key, name in model_names.items():
            if key in all_metrics:
                m = all_metrics[key]
                row = (
                    f"  {name:<20} "
                    f"{m.get('accuracy', 0):>10.4f} "
                    f"{m.get('precision', 0):>10.4f} "
                    f"{m.get('recall', 0):>10.4f} "
                    f"{m.get('f1_score', 0):>10.4f} "
                    f"{m.get('roc_auc', 0):>10.4f}"
                )
                logger.info(row)

    def get_best_model_name(self, all_metrics: Dict, metric: str = 'f1_score') -> str:
        """Return name of best performing model"""
        best_name = max(all_metrics, key=lambda k: all_metrics[k].get(metric, 0))
        return best_name

    def save_all(self):
        """Save all models to disk"""
        logger.subsection("Saving All Models")
        self.rf.save()
        self.xgb.save()
        self.svm.save()
        logger.success("All models saved!")

    def load_all(self):
        """Load all models from disk"""
        logger.subsection("Loading All Models")
        self.rf.load()
        self.xgb.load()
        self.svm.load()
        logger.success("All models loaded!")


def run_classification_pipeline(X_train: pd.DataFrame, X_test: pd.DataFrame,
                                 y_train: pd.Series, y_test: pd.Series,
                                 feature_names: list = None,
                                 save_models: bool = True,
                                 verbose: bool = True) -> Dict:
    """
    Complete classification pipeline: train all models + evaluate

    Args:
        X_train, X_test: Features
        y_train, y_test: Labels
        feature_names: Feature names for importance analysis
        save_models: Whether to save trained models
        verbose: Print progress

    Returns:
        Dictionary with all results
    """
    ensemble = EnsembleModel()

    # Train all models
    ensemble.train_all(X_train, y_train,
                       feature_names=feature_names, verbose=verbose)

    # Evaluate all models
    all_metrics = ensemble.evaluate_all(X_test, y_test, verbose=verbose)

    # Feature importance from RF
    rf_importance = ensemble.rf.get_feature_importance()
    xgb_importance = ensemble.xgb.get_feature_importance()

    if verbose:
        logger.subsection("Top 5 Important Features (Random Forest)")
        for _, row in rf_importance.head(5).iterrows():
            logger.info(f"  {row['feature']:<35} {row['importance']:.4f}")

    # Save if requested
    if save_models:
        ensemble.save_all()

    return {
        'ensemble': ensemble,
        'all_metrics': all_metrics,
        'rf_importance': rf_importance,
        'xgb_importance': xgb_importance
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

    from src.data.data_loader import WaterPotabilityLoader
    from src.data.preprocessing_classification import preprocess_classification_data

    # Load and preprocess
    loader = WaterPotabilityLoader()
    data, _ = loader.load(verbose=False)
    X, y = loader.get_features_and_target()

    results = preprocess_classification_data(X, y, apply_smote=False,
                                             save_results=False, verbose=False)

    # Run full pipeline
    pipeline_results = run_classification_pipeline(
        results['X_train'], results['X_test'],
        results['y_train'], results['y_test'],
        feature_names=results['feature_names'],
        save_models=True,
        verbose=True
    )

    print("\n✓ Ensemble classification pipeline test passed!")
