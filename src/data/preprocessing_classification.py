"""
============================================================================
Classification Data Preprocessing Pipeline
Handles water potability dataset preprocessing including:
- Missing value imputation
- Outlier detection and removal
- Feature scaling
- Class balancing (SMOTE)
- Train-test splitting
============================================================================
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import KNNImputer, SimpleImputer
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.pipeline import Pipeline as ImbPipeline
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    CLASSIFICATION_CONFIG,
    PROCESSED_DATA_DIR,
    MODEL_PATHS,
    RANDOM_SEED
)
from utils.logger import get_logger
from utils.file_handler import FileHandler
from utils.helpers import (
    detect_outliers_iqr,
    detect_outliers_zscore,
    get_missing_value_summary,
    calculate_class_weights,
    set_random_seeds
)

logger = get_logger()


class ClassificationPreprocessor:
    """
    Preprocessing pipeline for water potability classification
    
    Steps:
    1. Handle missing values
    2. Detect and remove outliers
    3. Scale features
    4. Balance classes (SMOTE)
    5. Split data
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize preprocessor
        
        Args:
            config: Configuration dictionary (uses default if None)
        """
        self.config = config or CLASSIFICATION_CONFIG
        self.scaler = None
        self.imputer = None
        self.feature_names = None
        self.class_weights = None
        
        # Statistics for reporting
        self.stats = {
            'original_samples': 0,
            'missing_values_imputed': 0,
            'outliers_removed': 0,
            'final_samples': 0,
            'smote_samples_added': 0,
            'class_distribution_before': {},
            'class_distribution_after': {}
        }
        
        set_random_seeds(RANDOM_SEED)
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series, 
                     verbose: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Fit preprocessor and transform data
        
        Args:
            X: Features DataFrame
            y: Target Series
            verbose: Whether to print progress
        
        Returns:
            Tuple of (X_processed, y_processed)
        """
        if verbose:
            logger.section("CLASSIFICATION PREPROCESSING")
        
        self.feature_names = list(X.columns)
        self.stats['original_samples'] = len(X)
        self.stats['class_distribution_before'] = dict(y.value_counts())
        
        # Step 1: Handle missing values
        X_imputed, y = self._handle_missing_values(X, y, verbose)
        
        # Step 2: Remove outliers
        X_no_outliers, y = self._remove_outliers(X_imputed, y, verbose)
        
        # Step 3: Scale features
        X_scaled = self._scale_features(X_no_outliers, fit=True, verbose=verbose)
        
        # Step 4: Calculate class weights (before SMOTE)
        self.class_weights = calculate_class_weights(y.values)
        
        self.stats['final_samples'] = len(X_scaled)
        
        if verbose:
            self._print_summary()
        
        return X_scaled, y
    
    def transform(self, X: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
        """
        Transform new data using fitted preprocessor
        
        Args:
            X: Features DataFrame
            verbose: Whether to print progress
        
        Returns:
            Transformed DataFrame
        """
        if self.scaler is None or self.imputer is None:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
        
        # Impute missing values
        X_imputed = pd.DataFrame(
            self.imputer.transform(X),
            columns=self.feature_names,
            index=X.index
        )
        
        # Scale features
        X_scaled = self._scale_features(X_imputed, fit=False, verbose=verbose)
        
        return X_scaled
    
    def _handle_missing_values(self, X: pd.DataFrame, y: pd.Series, 
                               verbose: bool) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Handle missing values using imputation
        
        Args:
            X: Features
            y: Target
            verbose: Print progress
        
        Returns:
            Tuple of (X_imputed, y)
        """
        if verbose:
            logger.subsection("Step 1: Handling Missing Values")
        
        # Get missing value summary
        missing_summary = get_missing_value_summary(X)
        
        if len(missing_summary) > 0:
            if verbose:
                logger.info(f"  Missing values detected in {len(missing_summary)} features:")
                for col, row in missing_summary.iterrows():
                    logger.info(f"    {col}: {int(row['Missing_Count'])} ({row['Missing_Percent']:.1f}%)")
            
            # Choose imputation method
            method = self.config.get('handle_missing', 'knn')
            
            if method == 'knn':
                n_neighbors = self.config.get('knn_neighbors', 5)
                self.imputer = KNNImputer(n_neighbors=n_neighbors)
                if verbose:
                    logger.info(f"  Using KNN imputation (k={n_neighbors})")
            
            elif method == 'mean':
                self.imputer = SimpleImputer(strategy='mean')
                if verbose:
                    logger.info("  Using mean imputation")
            
            elif method == 'median':
                self.imputer = SimpleImputer(strategy='median')
                if verbose:
                    logger.info("  Using median imputation")
            
            else:
                raise ValueError(f"Unknown imputation method: {method}")
            
            # Fit and transform
            X_imputed = pd.DataFrame(
                self.imputer.fit_transform(X),
                columns=self.feature_names,
                index=X.index
            )
            
            self.stats['missing_values_imputed'] = int(missing_summary['Missing_Count'].sum())
            
            if verbose:
                logger.success(f"  Imputed {self.stats['missing_values_imputed']} missing values")
        
        else:
            X_imputed = X.copy()
            self.imputer = None
            if verbose:
                logger.success("  No missing values detected")
        
        return X_imputed, y
    
    def _remove_outliers(self, X: pd.DataFrame, y: pd.Series, 
                        verbose: bool) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Detect and remove outliers
        
        Args:
            X: Features
            y: Target
            verbose: Print progress
        
        Returns:
            Tuple of (X_no_outliers, y_no_outliers)
        """
        if verbose:
            logger.subsection("Step 2: Detecting and Removing Outliers")
        
        method = self.config.get('outlier_method', 'iqr')
        threshold = self.config.get('outlier_threshold', 1.5)
        
        if method == 'none':
            if verbose:
                logger.info("  Outlier removal disabled")
            return X, y
        
        # Detect outliers for each feature
        outlier_masks = []
        
        for col in X.columns:
            if method == 'iqr':
                mask, stats = detect_outliers_iqr(X[col].values, threshold=threshold)
            elif method == 'zscore':
                mask, stats = detect_outliers_zscore(X[col].values, threshold=threshold)
            else:
                raise ValueError(f"Unknown outlier method: {method}")
            
            outlier_masks.append(mask)
            
            if verbose and stats['n_outliers'] > 0:
                logger.info(f"  {col}: {stats['n_outliers']} outliers ({stats['outlier_percent']:.1f}%)")
        
        # Combine masks (remove row if ANY feature is outlier)
        combined_mask = np.any(outlier_masks, axis=0)
        n_outliers = np.sum(combined_mask)
        
        if n_outliers > 0:
            # Remove outliers
            X_clean = X[~combined_mask].copy()
            y_clean = y[~combined_mask].copy()
            
            self.stats['outliers_removed'] = n_outliers
            
            if verbose:
                logger.success(f"  Removed {n_outliers} outlier samples ({n_outliers/len(X)*100:.1f}%)")
                logger.info(f"  Remaining samples: {len(X_clean)}")
        
        else:
            X_clean = X.copy()
            y_clean = y.copy()
            if verbose:
                logger.success("  No outliers detected")
        
        return X_clean, y_clean
    
    def _scale_features(self, X: pd.DataFrame, fit: bool = True, 
                       verbose: bool = False) -> pd.DataFrame:
        """
        Scale features
        
        Args:
            X: Features
            fit: Whether to fit scaler
            verbose: Print progress
        
        Returns:
            Scaled features
        """
        if fit:
            if verbose:
                logger.subsection("Step 3: Scaling Features")
            
            method = self.config.get('scaling_method', 'standard')
            
            if method == 'standard':
                self.scaler = StandardScaler()
                scaler_name = "StandardScaler"
            elif method == 'minmax':
                self.scaler = MinMaxScaler()
                scaler_name = "MinMaxScaler"
            elif method == 'robust':
                self.scaler = RobustScaler()
                scaler_name = "RobustScaler"
            else:
                raise ValueError(f"Unknown scaling method: {method}")
            
            if verbose:
                logger.info(f"  Using {scaler_name}")
            
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=self.feature_names,
                index=X.index
            )
            
            if verbose:
                logger.success("  Features scaled successfully")
        
        else:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=self.feature_names,
                index=X.index
            )
        
        return X_scaled
    
    def apply_smote(self, X: pd.DataFrame, y: pd.Series, 
                   verbose: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply SMOTE for class balancing
        
        Args:
            X: Features
            y: Target
            verbose: Print progress
        
        Returns:
            Tuple of (X_balanced, y_balanced)
        """
        if not self.config.get('apply_smote', True):
            if verbose:
                logger.info("  SMOTE disabled")
            return X, y

        if not SMOTE_AVAILABLE:
            logger.warning("  SMOTE skipped - install: pip install imbalanced-learn")
            return X, y

        if verbose:
            logger.subsection("Step 4: Balancing Classes (SMOTE)")
        
        # Original distribution
        class_counts_before = dict(y.value_counts().sort_index())
        
        if verbose:
            logger.info("  Class distribution before SMOTE:")
            for cls, count in class_counts_before.items():
                logger.info(f"    Class {cls}: {count} ({count/len(y)*100:.1f}%)")
        
        # Apply SMOTE
        sampling_strategy = self.config.get('smote_sampling_strategy', 'auto')
        k_neighbors = self.config.get('smote_k_neighbors', 5)
        
        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=k_neighbors,
            random_state=RANDOM_SEED
        )
        
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        # Convert back to DataFrame/Series
        X_balanced = pd.DataFrame(X_resampled, columns=self.feature_names)
        y_balanced = pd.Series(y_resampled, name=y.name)
        
        # New distribution
        class_counts_after = dict(y_balanced.value_counts().sort_index())
        self.stats['class_distribution_after'] = class_counts_after
        self.stats['smote_samples_added'] = len(y_balanced) - len(y)
        
        if verbose:
            logger.info("  Class distribution after SMOTE:")
            for cls, count in class_counts_after.items():
                logger.info(f"    Class {cls}: {count} ({count/len(y_balanced)*100:.1f}%)")
            logger.success(f"  Added {self.stats['smote_samples_added']} synthetic samples")
        
        return X_balanced, y_balanced
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, 
                  test_size: float = None,
                  verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, 
                                                  pd.Series, pd.Series]:
        """
        Split data into train and test sets
        
        Args:
            X: Features
            y: Target
            test_size: Test set proportion
            verbose: Print progress
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        test_size = test_size or self.config.get('test_size', 0.2)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=RANDOM_SEED,
            stratify=y
        )
        
        if verbose:
            logger.subsection("Train-Test Split")
            logger.info(f"  Train samples: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
            logger.info(f"  Test samples: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
            
            # Class distribution in train
            train_dist = dict(y_train.value_counts().sort_index())
            logger.info("  Train class distribution:")
            for cls, count in train_dist.items():
                logger.info(f"    Class {cls}: {count} ({count/len(y_train)*100:.1f}%)")
        
        return X_train, X_test, y_train, y_test
    
    def _print_summary(self):
        """Print preprocessing summary"""
        logger.section("PREPROCESSING SUMMARY")
        
        logger.info(f"  Original samples: {self.stats['original_samples']}")
        logger.info(f"  Missing values imputed: {self.stats['missing_values_imputed']}")
        logger.info(f"  Outliers removed: {self.stats['outliers_removed']}")
        logger.info(f"  Final samples: {self.stats['final_samples']}")
        
        logger.info("\n  Class distribution (before balancing):")
        for cls, count in self.stats['class_distribution_before'].items():
            pct = count / self.stats['original_samples'] * 100
            logger.info(f"    Class {cls}: {count} ({pct:.1f}%)")
        
        if self.class_weights:
            logger.info("\n  Calculated class weights:")
            for cls, weight in self.class_weights.items():
                logger.info(f"    Class {cls}: {weight:.4f}")
    
    def save(self, output_dir: Path = PROCESSED_DATA_DIR):
        """
        Save preprocessor components
        
        Args:
            output_dir: Directory to save components
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save scaler
        if self.scaler is not None:
            FileHandler.save_model_joblib(
                self.scaler,
                MODEL_PATHS['scaler_classification']
            )
        
        # Save imputer
        if self.imputer is not None:
            FileHandler.save_model_joblib(
                self.imputer,
                output_dir / 'imputer.pkl'
            )
        
        # Save statistics
        FileHandler.save_json(
            self.stats,
            output_dir / 'preprocessing_stats.json'
        )
        
        # Save feature names
        FileHandler.save_json(
            {'features': self.feature_names},
            output_dir / 'feature_names.json'
        )
        
        logger.success("Preprocessor saved successfully")


def preprocess_classification_data(X: pd.DataFrame, y: pd.Series,
                                   apply_smote: bool = True,
                                   save_results: bool = True,
                                   verbose: bool = True) -> Dict[str, Any]:
    """
    Complete preprocessing pipeline for classification data
    
    Args:
        X: Features DataFrame
        y: Target Series
        apply_smote: Whether to apply SMOTE
        save_results: Whether to save preprocessed data
        verbose: Print progress
    
    Returns:
        Dictionary containing all preprocessed components
    """
    # Initialize preprocessor
    preprocessor = ClassificationPreprocessor()
    
    # Fit and transform
    X_processed, y_processed = preprocessor.fit_transform(X, y, verbose=verbose)
    
    # Split FIRST (stratified 80/20), so the test set holds only real samples
    X_train, X_test, y_train, y_test = preprocessor.split_data(
        X_processed, y_processed, verbose=verbose
    )

    # Apply SMOTE to the TRAINING set only (prevents synthetic samples leaking into test)
    if apply_smote:
        X_train, y_train = preprocessor.apply_smote(
            X_train, y_train, verbose=verbose
        )
    
    # Save results
    if save_results:
        logger.subsection("Saving Preprocessed Data")
        
        FileHandler.save_dataframe(X_train, PROCESSED_DATA_DIR / 'classification_X_train.csv')
        FileHandler.save_dataframe(X_test, PROCESSED_DATA_DIR / 'classification_X_test.csv')
        FileHandler.save_dataframe(
            pd.DataFrame({'Potability': y_train}),
            PROCESSED_DATA_DIR / 'classification_y_train.csv'
        )
        FileHandler.save_dataframe(
            pd.DataFrame({'Potability': y_test}),
            PROCESSED_DATA_DIR / 'classification_y_test.csv'
        )
        
        preprocessor.save()
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'preprocessor': preprocessor,
        'scaler': preprocessor.scaler,
        'class_weights': preprocessor.class_weights,
        'feature_names': preprocessor.feature_names,
        'stats': preprocessor.stats
    }


if __name__ == "__main__":
    # Test preprocessing pipeline
    from data.data_loader import WaterPotabilityLoader
    
    # Load data
    loader = WaterPotabilityLoader()
    data, success = loader.load(verbose=False)
    
    if success:
        X, y = loader.get_features_and_target()
        
        # Preprocess
        results = preprocess_classification_data(X, y, verbose=True)
        
        print("\n" + "="*70)
        print("✓ Classification preprocessing complete!")
        print(f"  Train samples: {len(results['X_train'])}")
        print(f"  Test samples: {len(results['X_test'])}")
        print("="*70)
