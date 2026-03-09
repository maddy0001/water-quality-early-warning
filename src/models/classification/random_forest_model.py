"""
============================================================================
Random Forest Classification Model
Water Potability Prediction with Feature Importance Analysis
============================================================================
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import CLASSIFICATION_CONFIG, MODEL_PATHS, RANDOM_SEED
from utils.logger import get_logger
from utils.file_handler import FileHandler

logger = get_logger()


class RandomForestModel:
    """
    Random Forest Classifier for water potability prediction
    
    Features:
    - Configurable hyperparameters
    - Cross-validation support
    - Feature importance analysis
    - Model save/load
    """

    def __init__(self, config: Dict = None):
        self.config = config or CLASSIFICATION_CONFIG
        self.model = None
        self.feature_names = None
        self.metrics = {}
        self._build_model()

    def _build_model(self):
        """Build Random Forest model"""
        self.model = RandomForestClassifier(
            n_estimators=self.config.get('rf_n_estimators', 200),
            max_depth=self.config.get('rf_max_depth', 15),
            min_samples_split=self.config.get('rf_min_samples_split', 5),
            min_samples_leaf=self.config.get('rf_min_samples_leaf', 2),
            class_weight=self.config.get('rf_class_weight', 'balanced'),
            n_jobs=self.config.get('rf_n_jobs', -1),
            random_state=RANDOM_SEED
        )
        logger.info("  Random Forest model built")
        logger.info(f"    Estimators: {self.config.get('rf_n_estimators', 200)}")
        logger.info(f"    Max depth: {self.config.get('rf_max_depth', 15)}")

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              feature_names: list = None, verbose: bool = True) -> Dict:
        """
        Train the Random Forest model
        
        Args:
            X_train: Training features
            y_train: Training labels
            feature_names: List of feature names
            verbose: Print progress

        Returns:
            Training metrics dictionary
        """
        if verbose:
            logger.subsection("Training Random Forest")

        self.feature_names = feature_names or (
            list(X_train.columns) if hasattr(X_train, 'columns') else None
        )

        import time
        start = time.time()
        self.model.fit(X_train, y_train)
        train_time = time.time() - start

        # Training metrics
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
        """
        Evaluate model on test set
        
        Returns:
            Evaluation metrics dictionary
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

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
            logger.subsection("Random Forest Evaluation")
            logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
            logger.info(f"  Precision: {metrics['precision']:.4f}")
            logger.info(f"  Recall:    {metrics['recall']:.4f}")
            logger.info(f"  F1-Score:  {metrics['f1_score']:.4f}")
            logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")

        return metrics

    def cross_validate(self, X: pd.DataFrame, y: pd.Series,
                       cv_folds: int = 5, verbose: bool = True) -> Dict:
        """
        Perform stratified k-fold cross-validation
        
        Returns:
            Cross-validation metrics
        """
        if verbose:
            logger.info(f"  Running {cv_folds}-fold cross-validation...")

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)

        cv_results = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            scores = cross_val_score(
                self.model, X, y, cv=cv,
                scoring=metric, n_jobs=-1
            )
            cv_results[f'cv_{metric}_mean'] = scores.mean()
            cv_results[f'cv_{metric}_std'] = scores.std()

            if verbose:
                logger.info(f"  CV {metric}: {scores.mean():.4f} ± {scores.std():.4f}")

        self.metrics.update(cv_results)
        return cv_results

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores
        
        Returns:
            DataFrame with feature importances ranked
        """
        if self.model is None:
            raise ValueError("Model not trained.")

        importances = self.model.feature_importances_
        names = self.feature_names or [f'Feature_{i}' for i in range(len(importances))]

        importance_df = pd.DataFrame({
            'feature': names,
            'importance': importances,
            'std': np.std([
                tree.feature_importances_ for tree in self.model.estimators_
            ], axis=0)
        }).sort_values('importance', ascending=False).reset_index(drop=True)

        return importance_df

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels"""
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities"""
        return self.model.predict_proba(X)

    def save(self, filepath: Path = None):
        """Save model to disk"""
        filepath = filepath or MODEL_PATHS['random_forest']
        FileHandler.save_model_joblib(self.model, filepath)

    def load(self, filepath: Path = None):
        """Load model from disk"""
        filepath = filepath or MODEL_PATHS['random_forest']
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

    rf = RandomForestModel()
    rf.train(results['X_train'], results['y_train'],
             feature_names=results['feature_names'])
    metrics = rf.evaluate(results['X_test'], results['y_test'])

    importance_df = rf.get_feature_importance()
    print("\nTop Features:")
    print(importance_df.head())
    print("\n✓ Random Forest model test passed!")
