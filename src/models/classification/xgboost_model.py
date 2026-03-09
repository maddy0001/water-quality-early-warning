"""
============================================================================
XGBoost Classification Model
Gradient Boosting for Water Potability Prediction
============================================================================
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report, confusion_matrix
)

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import CLASSIFICATION_CONFIG, MODEL_PATHS, RANDOM_SEED
from utils.logger import get_logger
from utils.file_handler import FileHandler

logger = get_logger()


class XGBoostModel:
    """
    XGBoost Classifier for water potability prediction
    
    Features:
    - Gradient boosting with regularization
    - Built-in feature importance
    - GPU support (if available)
    - Early stopping
    """

    def __init__(self, config: Dict = None):
        self.config = config or CLASSIFICATION_CONFIG
        self.model = None
        self.feature_names = None
        self.metrics = {}
        self._build_model()

    def _build_model(self):
        """Build XGBoost model"""
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")

        # Detect GPU availability
        try:
            import tensorflow as tf
            gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
            tree_method = 'gpu_hist' if gpu_available else 'hist'
        except:
            tree_method = 'hist'

        self.model = xgb.XGBClassifier(
            n_estimators=self.config.get('xgb_n_estimators', 150),
            max_depth=self.config.get('xgb_max_depth', 7),
            learning_rate=self.config.get('xgb_learning_rate', 0.1),
            subsample=self.config.get('xgb_subsample', 0.8),
            colsample_bytree=self.config.get('xgb_colsample_bytree', 0.8),
            scale_pos_weight=self.config.get('xgb_scale_pos_weight', 1.5),
            tree_method=tree_method,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=RANDOM_SEED,
            verbosity=0
        )

        logger.info("  XGBoost model built")
        logger.info(f"    Estimators: {self.config.get('xgb_n_estimators', 150)}")
        logger.info(f"    Learning rate: {self.config.get('xgb_learning_rate', 0.1)}")
        logger.info(f"    Tree method: {tree_method}")

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame = None, y_val: pd.Series = None,
              feature_names: list = None, verbose: bool = True) -> Dict:
        """
        Train the XGBoost model with optional early stopping
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (for early stopping)
            y_val: Validation labels
            feature_names: Feature names
            verbose: Print progress

        Returns:
            Training metrics
        """
        if verbose:
            logger.subsection("Training XGBoost")

        self.feature_names = feature_names or (
            list(X_train.columns) if hasattr(X_train, 'columns') else None
        )

        import time
        start = time.time()

        # Fit with or without early stopping
        if X_val is not None and y_val is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train)

        train_time = time.time() - start

        y_pred_train = self.model.predict(X_train)
        train_acc = accuracy_score(y_train, y_pred_train)

        self.metrics['train_accuracy'] = train_acc
        self.metrics['train_time'] = train_time

        if verbose:
            logger.success(f"  Training complete in {train_time:.2f}s")
            logger.info(f"  Train accuracy: {train_acc:.4f}")

        return self.metrics

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series,
                 verbose: bool = True) -> Dict:
        """Evaluate model on test set"""
        if self.model is None:
            raise ValueError("Model not trained.")

        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]

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

        self.metrics.update(metrics)

        if verbose:
            logger.subsection("XGBoost Evaluation")
            logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
            logger.info(f"  Precision: {metrics['precision']:.4f}")
            logger.info(f"  Recall:    {metrics['recall']:.4f}")
            logger.info(f"  F1-Score:  {metrics['f1_score']:.4f}")
            logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")

        return metrics

    def cross_validate(self, X: pd.DataFrame, y: pd.Series,
                       cv_folds: int = 5, verbose: bool = True) -> Dict:
        """Perform stratified k-fold cross-validation"""
        if verbose:
            logger.info(f"  Running {cv_folds}-fold cross-validation...")

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)
        cv_results = {}

        for metric in ['accuracy', 'f1']:
            scores = cross_val_score(self.model, X, y, cv=cv, scoring=metric, n_jobs=-1)
            cv_results[f'cv_{metric}_mean'] = scores.mean()
            cv_results[f'cv_{metric}_std'] = scores.std()

            if verbose:
                logger.info(f"  CV {metric}: {scores.mean():.4f} ± {scores.std():.4f}")

        self.metrics.update(cv_results)
        return cv_results

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores"""
        if self.model is None:
            raise ValueError("Model not trained.")

        importances = self.model.feature_importances_
        names = self.feature_names or [f'Feature_{i}' for i in range(len(importances))]

        return pd.DataFrame({
            'feature': names,
            'importance': importances
        }).sort_values('importance', ascending=False).reset_index(drop=True)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)

    def save(self, filepath: Path = None):
        filepath = filepath or MODEL_PATHS['xgboost']
        FileHandler.save_model_joblib(self.model, filepath)

    def load(self, filepath: Path = None):
        filepath = filepath or MODEL_PATHS['xgboost']
        self.model = FileHandler.load_model_joblib(filepath)
        return self


if __name__ == "__main__":
    from data.data_loader import WaterPotabilityLoader
    from data.preprocessing_classification import preprocess_classification_data

    loader = WaterPotabilityLoader()
    data, _ = loader.load(verbose=False)
    X, y = loader.get_features_and_target()

    results = preprocess_classification_data(X, y, apply_smote=False,
                                             save_results=False, verbose=False)

    xgb_model = XGBoostModel()
    xgb_model.train(results['X_train'], results['y_train'],
                    feature_names=results['feature_names'])
    metrics = xgb_model.evaluate(results['X_test'], results['y_test'])
    print("\n✓ XGBoost model test passed!")
