"""
============================================================================
SVM Classification Model
Support Vector Machine for Water Potability Prediction
============================================================================
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
from sklearn.svm import SVC
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


class SVMModel:
    """
    SVM Classifier for water potability prediction
    
    Features:
    - RBF kernel (best for non-linear data)
    - Probability calibration for soft voting
    - Class weight balancing
    """

    def __init__(self, config: Dict = None):
        self.config = config or CLASSIFICATION_CONFIG
        self.model = None
        self.feature_names = None
        self.metrics = {}
        self._build_model()

    def _build_model(self):
        """Build SVM model"""
        self.model = SVC(
            C=self.config.get('svm_C', 10),
            kernel=self.config.get('svm_kernel', 'rbf'),
            gamma=self.config.get('svm_gamma', 'scale'),
            class_weight=self.config.get('svm_class_weight', 'balanced'),
            probability=True,   # Required for soft ensemble voting
            random_state=RANDOM_SEED
        )

        logger.info("  SVM model built")
        logger.info(f"    C: {self.config.get('svm_C', 10)}")
        logger.info(f"    Kernel: {self.config.get('svm_kernel', 'rbf')}")
        logger.info(f"    Gamma: {self.config.get('svm_gamma', 'scale')}")

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              feature_names: list = None, verbose: bool = True) -> Dict:
        """
        Train the SVM model
        Note: SVM is slower than tree-based models; may take 2-4 minutes
        """
        if verbose:
            logger.subsection("Training SVM")
            logger.info("  Note: SVM training may take 2-5 minutes...")

        self.feature_names = feature_names or (
            list(X_train.columns) if hasattr(X_train, 'columns') else None
        )

        import time
        start = time.time()
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
            logger.subsection("SVM Evaluation")
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
            logger.info("  Note: SVM CV may take several minutes...")

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

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)

    def save(self, filepath: Path = None):
        filepath = filepath or MODEL_PATHS['svm']
        FileHandler.save_model_joblib(self.model, filepath)

    def load(self, filepath: Path = None):
        filepath = filepath or MODEL_PATHS['svm']
        self.model = FileHandler.load_model_joblib(filepath)
        return self
