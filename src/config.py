"""
============================================================================
AI-Based Water Contamination Early Warning System
Master Configuration File
============================================================================
"""

import os
from pathlib import Path
import numpy as np

# ============================================================================
# PROJECT PATHS
# ============================================================================
BASE_DIR = Path(__file__).parent.parent  # Go up from src/ to project root
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
LOGS_DIR = OUTPUTS_DIR / "logs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
METRICS_DIR = OUTPUTS_DIR / "metrics"

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, 
                  OUTPUTS_DIR, LOGS_DIR, PLOTS_DIR, METRICS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATASET PATHS
# ============================================================================
# Classification Dataset
WATER_POTABILITY_PATH = RAW_DATA_DIR / "water_potability.csv"

# Forecasting Dataset
WATER_DATASET_MAT_PATH = RAW_DATA_DIR / "water_dataset.mat"

# ============================================================================
# GENERAL SETTINGS
# ============================================================================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# GPU Settings
USE_GPU = True  # Will auto-detect and fallback to CPU if not available
GPU_MEMORY_LIMIT = 4096  # MB

# Logging
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
VERBOSE = True

# ============================================================================
# CLASSIFICATION PIPELINE SETTINGS
# ============================================================================
CLASSIFICATION_CONFIG = {
    # Data Preprocessing
    'test_size': 0.2,
    'validation_split': 0.1,
    'handle_missing': 'knn',  # 'knn', 'mean', 'median', 'drop'
    'knn_neighbors': 5,
    'outlier_method': 'iqr',  # 'iqr', 'zscore', 'isolation_forest'
    'outlier_threshold': 1.5,
    'scaling_method': 'standard',  # 'standard', 'minmax', 'robust'
    
    # Class Balancing
    'apply_smote': True,
    'smote_sampling_strategy': 'auto',
    'smote_k_neighbors': 5,
    
    # Feature Selection
    'feature_selection': False,
    'n_features': 7,
    
    # Cross-Validation
    'cv_folds': 5,
    'cv_scoring': 'f1',
    
    # Random Forest
    'rf_n_estimators': 200,
    'rf_max_depth': 15,
    'rf_min_samples_split': 5,
    'rf_min_samples_leaf': 2,
    'rf_class_weight': 'balanced',
    'rf_n_jobs': -1,
    
    # XGBoost
    'xgb_n_estimators': 150,
    'xgb_max_depth': 7,
    'xgb_learning_rate': 0.1,
    'xgb_subsample': 0.8,
    'xgb_colsample_bytree': 0.8,
    'xgb_scale_pos_weight': 1.5,
    
    # SVM
    'svm_C': 10,
    'svm_kernel': 'rbf',
    'svm_gamma': 'scale',
    'svm_class_weight': 'balanced',
    
    # Ensemble
    'ensemble_voting': 'soft',  # 'soft' or 'hard'
    'ensemble_weights': [0.35, 0.40, 0.25],  # [RF, XGB, SVM]
}

# ============================================================================
# FORECASTING PIPELINE SETTINGS
# ============================================================================
FORECASTING_CONFIG = {
    # Sequence Generation
    'lookback_window': 30,  # Use 30 days to predict next day
    'forecast_horizon': 1,   # Predict 1 day ahead
    'stride': 1,
    
    # Data Split
    'train_ratio': 0.6,
    'val_ratio': 0.2,
    'test_ratio': 0.2,
    
    # Normalization
    'scaling_method': 'minmax',  # 'minmax' or 'standard'
    'feature_range': (0, 1),
    
    # Bi-LSTM with Attention
    'lstm_units': [128, 64],
    'lstm_dropout': 0.3,
    'lstm_recurrent_dropout': 0.2,
    'lstm_activation': 'tanh',
    'attention_units': 64,
    'dense_units': [32, 16],
    'output_activation': 'linear',
    
    # GRU with Attention
    'gru_units': [128, 64],
    'gru_dropout': 0.3,
    'gru_recurrent_dropout': 0.2,
    
    # Training
    'epochs': 100,
    'batch_size': 32,
    'learning_rate': 0.001,
    'optimizer': 'adam',
    'loss': 'mse',
    'early_stopping_patience': 15,
    'reduce_lr_patience': 7,
    'reduce_lr_factor': 0.5,
    'min_lr': 1e-7,
    
    # Spatial
    'n_locations': 37,
    'spatial_graph': True,
}

# ============================================================================
# ANOMALY DETECTION SETTINGS
# ============================================================================
ANOMALY_CONFIG = {
    # Isolation Forest
    'if_n_estimators': 150,
    'if_contamination': 0.05,
    'if_max_samples': 'auto',
    'if_max_features': 1.0,
    
    # Local Outlier Factor
    'lof_n_neighbors': 20,
    'lof_contamination': 'auto',
    'lof_novelty': False,
    
    # One-Class SVM
    'ocsvm_kernel': 'rbf',
    'ocsvm_nu': 0.05,
    'ocsvm_gamma': 'scale',
    
    # Autoencoder
    'ae_encoding_dim': 8,
    'ae_hidden_layers': [16, 12],
    'ae_activation': 'relu',
    'ae_epochs': 50,
    'ae_batch_size': 32,
    'ae_contamination_threshold': 0.95,  # percentile
    
    # DBSCAN
    'dbscan_eps': 0.5,
    'dbscan_min_samples': 5,
    'dbscan_metric': 'euclidean',
    
    # Ensemble
    'anomaly_voting_threshold': 0.6,  # If 60% models agree = anomaly
}

# ============================================================================
# EARLY WARNING ENGINE SETTINGS
# ============================================================================
EARLY_WARNING_CONFIG = {
    # Alert Thresholds (risk scores 0-1)
    'safe_threshold': 0.3,
    'caution_threshold': 0.5,
    'warning_threshold': 0.7,
    # Above 0.7 = CRITICAL
    
    # Weight Configuration (sum = 1.0)
    'classification_weight': 0.3,
    'forecast_weight': 0.4,
    'anomaly_weight': 0.3,
    
    # Deviation Thresholds
    'forecast_deviation_threshold': 0.1,  # 10% deviation triggers alert
    'ph_safe_range': (6.5, 8.5),  # Safe pH range
    
    # Alert Rules
    'consecutive_warnings': 2,  # Escalate if 2+ consecutive warnings
    'critical_location_threshold': 0.5,  # If 50% locations critical
}

# ============================================================================
# EXPLAINABILITY SETTINGS
# ============================================================================
EXPLAINABILITY_CONFIG = {
    # SHAP
    'shap_n_samples': 100,
    'shap_check_additivity': False,
    
    # Feature Importance
    'permutation_n_repeats': 10,
    'permutation_scoring': 'f1',
}

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================
VISUALIZATION_CONFIG = {
    # General
    'figure_dpi': 100,
    'figure_size': (12, 8),
    'save_format': 'png',
    'color_palette': 'Set2',
    
    # Specific Plots
    'confusion_matrix_cmap': 'Blues',
    'heatmap_cmap': 'RdYlGn_r',
    'alert_colors': {
        'SAFE': '#2ecc71',
        'CAUTION': '#f39c12',
        'WARNING': '#e67e22',
        'CRITICAL': '#e74c3c'
    },
    
    # Font Sizes
    'title_fontsize': 16,
    'label_fontsize': 12,
    'tick_fontsize': 10,
}

# ============================================================================
# MODEL SAVE PATHS
# ============================================================================
MODEL_PATHS = {
    # Classification
    'random_forest': MODELS_DIR / 'classification' / 'random_forest.pkl',
    'xgboost': MODELS_DIR / 'classification' / 'xgboost.pkl',
    'svm': MODELS_DIR / 'classification' / 'svm.pkl',
    'ensemble': MODELS_DIR / 'classification' / 'ensemble.pkl',
    'scaler_classification': MODELS_DIR / 'classification' / 'scaler.pkl',
    
    # Forecasting
    'bilstm_attention': MODELS_DIR / 'forecasting' / 'bilstm_attention.h5',
    'gru_attention': MODELS_DIR / 'forecasting' / 'gru_attention.h5',
    'scaler_forecasting': MODELS_DIR / 'forecasting' / 'scaler.pkl',
    
    # Anomaly
    'isolation_forest': MODELS_DIR / 'anomaly' / 'isolation_forest.pkl',
    'lof': MODELS_DIR / 'anomaly' / 'lof.pkl',
    'ocsvm': MODELS_DIR / 'anomaly' / 'ocsvm.pkl',
    'autoencoder': MODELS_DIR / 'anomaly' / 'autoencoder.h5',
    'dbscan': MODELS_DIR / 'anomaly' / 'dbscan.pkl',
}

# ============================================================================
# FEATURE NAMES (for reference)
# ============================================================================
CLASSIFICATION_FEATURES = [
    'ph', 'Hardness', 'Solids', 'Chloramines', 
    'Sulfate', 'Conductivity', 'Organic_carbon', 
    'Trihalomethanes', 'Turbidity'
]

FORECASTING_FEATURES = [
    'Specific_conductance_max',
    'pH_max',
    'pH_min',
    'Specific_conductance_min',
    'Specific_conductance_mean',
    'Dissolved_oxygen_max',
    'Dissolved_oxygen_mean',
    'Dissolved_oxygen_min',
    'Temperature_mean',
    'Temperature_min',
    'Temperature_max'
]

# ============================================================================
# EXPERIMENT TRACKING
# ============================================================================
EXPERIMENT_CONFIG = {
    'track_experiments': True,
    'experiment_name': 'water_contamination_detection',
    'run_timestamp': True,
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def get_device():
    """Auto-detect GPU availability"""
    import tensorflow as tf
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus and USE_GPU:
        try:
            # Set memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Limit GPU memory
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=GPU_MEMORY_LIMIT)]
            )
            return "GPU"
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
            return "CPU"
    return "CPU"

def print_config():
    """Print current configuration summary"""
    print("=" * 70)
    print("CONFIGURATION SUMMARY".center(70))
    print("=" * 70)
    print(f"Random Seed: {RANDOM_SEED}")
    print(f"Device: {get_device()}")
    print(f"Classification Dataset: {WATER_POTABILITY_PATH.name}")
    print(f"Forecasting Dataset: {WATER_DATASET_MAT_PATH.name}")
    print(f"Output Directory: {OUTPUTS_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    print_config()
