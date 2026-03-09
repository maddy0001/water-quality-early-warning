"""
============================================================================
Forecasting Pipeline
Orchestrates data loading, preprocessing, BiLSTM + GRU training,
evaluation, and results saving for water quality pH forecasting.
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
from data.data_loader import WaterQualityForecastingLoader
from data.preprocessing_forecasting import preprocess_forecasting_data
from models.forecasting.bilstm_attention import BiLSTMAttentionModel
from models.forecasting.gru_attention import GRUAttentionModel

logger = get_logger()


def run_forecasting_pipeline(save_models: bool = True,
                              verbose: bool = True) -> Dict[str, Any]:
    """
    Full forecasting pipeline:
    1. Load UCI water quality dataset
    2. Preprocess + generate sequences
    3. Train BiLSTM with Attention
    4. Train GRU with Attention
    5. Compare and evaluate both
    6. Save models and metrics

    Returns:
        Complete results dictionary
    """
    set_random_seeds(RANDOM_SEED)
    pipeline_start = time.time()

    if verbose:
        logger.section("FORECASTING PIPELINE")

    # -------------------------------------------------------------------------
    # STEP 1: Load Data
    # -------------------------------------------------------------------------
    if verbose:
        logger.info("\n[1/5] Loading UCI Water Quality dataset...")

    loader = WaterQualityForecastingLoader()
    data, success = loader.load(verbose=False)

    if not success:
        logger.error("Failed to load forecasting dataset!")
        return {'success': False}

    logger.success(
        f"  Loaded: 705 days, 37 locations, 11 features"
    )

    # -------------------------------------------------------------------------
    # STEP 2: Preprocess + Sequences
    # -------------------------------------------------------------------------
    if verbose:
        logger.info("\n[2/5] Preprocessing & generating sequences...")

    prep = preprocess_forecasting_data(
        data['X_train'], data['X_test'],
        data['y_train'], data['y_test'],
        save_results=save_models,
        verbose=False
    )

    X_train = prep['X_train'].astype(np.float32)
    X_val   = prep['X_val'].astype(np.float32)
    X_test  = prep['X_test'].astype(np.float32)
    y_train = prep['y_train'].astype(np.float32)
    y_val   = prep['y_val'].astype(np.float32)
    y_test  = prep['y_test'].astype(np.float32)

    if verbose:
        logger.success(f"  Train: {X_train.shape} → Target: {y_train.shape}")
        logger.success(f"  Val:   {X_val.shape}   → Target: {y_val.shape}")
        logger.success(f"  Test:  {X_test.shape}  → Target: {y_test.shape}")

    input_shape = X_train.shape[1:]  # (30, 11)

    # -------------------------------------------------------------------------
    # STEP 3: Train BiLSTM + Attention
    # -------------------------------------------------------------------------
    if verbose:
        logger.info("\n[3/5] Training BiLSTM + Attention...")

    bilstm = BiLSTMAttentionModel()
    bilstm.build(input_shape=input_shape, verbose=verbose)
    bilstm.train(X_train, y_train, X_val, y_val, verbose=verbose)
    bilstm_metrics = bilstm.evaluate(X_test, y_test, verbose=verbose)

    # -------------------------------------------------------------------------
    # STEP 4: Train GRU + Attention
    # -------------------------------------------------------------------------
    if verbose:
        logger.info("\n[4/5] Training GRU + Attention...")

    gru = GRUAttentionModel()
    gru.build(input_shape=input_shape, verbose=verbose)
    gru.train(X_train, y_train, X_val, y_val, verbose=verbose)
    gru_metrics = gru.evaluate(X_test, y_test, verbose=verbose)

    # -------------------------------------------------------------------------
    # STEP 5: Compare + Save
    # -------------------------------------------------------------------------
    if verbose:
        logger.section("MODEL COMPARISON")
        _print_forecast_comparison(bilstm_metrics, gru_metrics)

    # Determine best model
    best_name = 'bilstm' if bilstm_metrics['rmse'] < gru_metrics['rmse'] else 'gru'

    if verbose:
        logger.success(f"  Best model: {best_name.upper()} (lower RMSE)")

    if save_models:
        logger.subsection("Saving Models")
        bilstm.save()
        gru.save()

        # Save metrics
        save_data = {
            'bilstm_attention': {k: v for k, v in bilstm_metrics.items()
                                 if k not in ('y_pred', 'y_true')},
            'gru_attention': {k: v for k, v in gru_metrics.items()
                              if k not in ('y_pred', 'y_true')},
            'best_model': best_name
        }
        FileHandler.save_json(save_data, METRICS_DIR / 'forecasting_metrics.json')
        logger.success("Models and metrics saved!")

    total_time = time.time() - pipeline_start

    logger.section("FORECASTING PIPELINE COMPLETE")
    logger.info(f"  Total time: {total_time/60:.1f} minutes")
    logger.info(f"  Best model: {best_name.upper()}")
    logger.info(f"  Best RMSE:  {min(bilstm_metrics['rmse'], gru_metrics['rmse']):.6f}")
    logger.info(f"  Best R²:    {max(bilstm_metrics['r2'], gru_metrics['r2']):.4f}")

    return {
        'success': True,
        'models': {'bilstm': bilstm, 'gru': gru},
        'metrics': {
            'bilstm': bilstm_metrics,
            'gru': gru_metrics
        },
        'best_model': best_name,
        'preprocessor': prep['preprocessor'],
        'data': {
            'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'y_train': y_train, 'y_val': y_val,   'y_test': y_test
        },
        'total_time': total_time
    }


def _print_forecast_comparison(bilstm_m: Dict, gru_m: Dict):
    """Print side-by-side model comparison"""
    logger.info(f"\n  {'Metric':<12} {'BiLSTM+Attention':>20} {'GRU+Attention':>20}")
    logger.info(f"  {'-'*52}")
    for metric in ['rmse', 'mae', 'mape', 'r2']:
        b_val = bilstm_m.get(metric, 0)
        g_val = gru_m.get(metric, 0)
        winner_b = " ←" if (
            (metric in ('rmse','mae','mape') and b_val < g_val) or
            (metric == 'r2' and b_val > g_val)
        ) else ""
        winner_g = " ←" if (
            (metric in ('rmse','mae','mape') and g_val < b_val) or
            (metric == 'r2' and g_val > b_val)
        ) else ""
        label = metric.upper()
        if metric == 'mape':
            logger.info(f"  {label:<12} {b_val:>18.2f}%{winner_b} {g_val:>18.2f}%{winner_g}")
        else:
            logger.info(f"  {label:<12} {b_val:>20.6f}{winner_b} {g_val:>20.6f}{winner_g}")


if __name__ == "__main__":
    results = run_forecasting_pipeline(save_models=True, verbose=True)

    if results['success']:
        print("\n" + "="*70)
        print("✓ Forecasting pipeline complete!")
        print(f"  BiLSTM RMSE: {results['metrics']['bilstm']['rmse']:.6f}")
        print(f"  GRU    RMSE: {results['metrics']['gru']['rmse']:.6f}")
        print(f"  Best model: {results['best_model'].upper()}")
        print(f"  Total time: {results['total_time']/60:.1f} minutes")
        print("="*70)
