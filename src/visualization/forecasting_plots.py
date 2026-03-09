"""
============================================================================
Forecasting Evaluation Plots
Time series predictions and error analysis
============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import get_logger

logger = get_logger()

plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
sns.set_style('whitegrid')


def plot_predictions_vs_actual(y_true: np.ndarray, y_pred: np.ndarray,
                               model_name: str = 'Model',
                               save_path: Path = None, show: bool = False):
    """
    Scatter plot: predictions vs actual values
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter(y_true, y_pred, alpha=0.5, s=20, edgecolors='k', linewidths=0.5)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    # Metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    ax.set_xlabel('Actual pH', fontweight='bold')
    ax.set_ylabel('Predicted pH', fontweight='bold')
    ax.set_title(f'Predictions vs Actual - {model_name}', fontweight='bold', fontsize=12)
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3)
    
    # Add metrics text box
    textstr = f'MAE = {mae:.4f}\nRMSE = {rmse:.4f}\nR² = {r2:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.success(f"  Saved: {save_path.name}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_time_series_forecast(y_true: np.ndarray, y_pred: np.ndarray,
                              n_show: int = 200,
                              model_name: str = 'Model',
                              save_path: Path = None, show: bool = False):
    """
    Time series plot showing predictions vs actual
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    
    # Limit to n_show points for readability
    if len(y_true) > n_show:
        y_true = y_true[:n_show]
        y_pred = y_pred[:n_show]
    
    x = np.arange(len(y_true))
    
    ax.plot(x, y_true, 'b-', label='Actual', linewidth=1.5, alpha=0.8)
    ax.plot(x, y_pred, 'r--', label='Predicted', linewidth=1.5, alpha=0.8)
    
    # Fill between for error visualization
    ax.fill_between(x, y_true, y_pred, alpha=0.2, color='gray')
    
    ax.set_xlabel('Time Step', fontweight='bold')
    ax.set_ylabel('pH Value', fontweight='bold')
    ax.set_title(f'Time Series Forecast - {model_name} (First {n_show} steps)', 
                fontweight='bold', fontsize=12)
    ax.legend(loc='best', frameon=True, fancybox=True)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.success(f"  Saved: {save_path.name}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray,
                  model_name: str = 'Model',
                  save_path: Path = None, show: bool = False):
    """
    Residual plot (errors vs predictions)
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Residuals vs predictions
    axes[0].scatter(y_pred, residuals, alpha=0.5, s=20, edgecolors='k', linewidths=0.5)
    axes[0].axhline(0, color='r', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Predicted Values', fontweight='bold')
    axes[0].set_ylabel('Residuals', fontweight='bold')
    axes[0].set_title('Residuals vs Predictions', fontweight='bold')
    axes[0].grid(alpha=0.3)
    
    # Residual distribution
    axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[1].axvline(0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Residuals', fontweight='bold')
    axes[1].set_ylabel('Frequency', fontweight='bold')
    axes[1].set_title('Residual Distribution', fontweight='bold')
    axes[1].grid(alpha=0.3)
    
    # Add stats
    mean_res = np.mean(residuals)
    std_res = np.std(residuals)
    textstr = f'Mean: {mean_res:.4f}\nStd: {std_res:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    axes[1].text(0.05, 0.95, textstr, transform=axes[1].transAxes,
                fontsize=10, verticalalignment='top', bbox=props)
    
    plt.suptitle(f'Residual Analysis - {model_name}', fontweight='bold', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.success(f"  Saved: {save_path.name}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_error_distribution(y_true: np.ndarray, y_pred: np.ndarray,
                           model_name: str = 'Model',
                           save_path: Path = None, show: bool = False):
    """
    Various error metrics visualization
    """
    errors = np.abs(y_true - y_pred)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Absolute errors histogram
    axes[0, 0].hist(errors, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0, 0].set_xlabel('Absolute Error', fontweight='bold')
    axes[0, 0].set_ylabel('Frequency', fontweight='bold')
    axes[0, 0].set_title('Absolute Error Distribution', fontweight='bold')
    axes[0, 0].axvline(np.mean(errors), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors):.4f}')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Cumulative error
    sorted_errors = np.sort(errors)
    cumulative = np.arange(1, len(sorted_errors)+1) / len(sorted_errors)
    axes[0, 1].plot(sorted_errors, cumulative, linewidth=2)
    axes[0, 1].set_xlabel('Absolute Error', fontweight='bold')
    axes[0, 1].set_ylabel('Cumulative Probability', fontweight='bold')
    axes[0, 1].set_title('Cumulative Error Distribution', fontweight='bold')
    axes[0, 1].grid(alpha=0.3)
    
    # Error over time
    axes[1, 0].plot(errors, alpha=0.6)
    axes[1, 0].set_xlabel('Sample Index', fontweight='bold')
    axes[1, 0].set_ylabel('Absolute Error', fontweight='bold')
    axes[1, 0].set_title('Error Over Time', fontweight='bold')
    axes[1, 0].grid(alpha=0.3)
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(errors, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot (Normality Check)', fontweight='bold')
    axes[1, 1].grid(alpha=0.3)
    
    plt.suptitle(f'Error Analysis - {model_name}', fontweight='bold', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.success(f"  Saved: {save_path.name}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_model_comparison_forecasting(models_results: Dict[str, Dict],
                                     save_path: Path = None, show: bool = False):
    """
    Compare forecasting models performance
    """
    metrics = ['rmse', 'mae', 'mape', 'r2']
    metric_names = ['RMSE', 'MAE', 'MAPE (%)', 'R²']
    
    model_names = list(models_results.keys())
    n_models = len(model_names)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx]
        
        values = [models_results[model].get(metric, 0) for model in model_names]
        
        bars = ax.bar(range(n_models), values, color=colors[:n_models], alpha=0.8)
        ax.set_xticks(range(n_models))
        ax.set_xticklabels(model_names, rotation=15, ha='right')
        ax.set_ylabel(metric_name, fontweight='bold')
        ax.set_title(f'{metric_name} Comparison', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Forecasting Models Comparison', fontweight='bold', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.success(f"  Saved: {save_path.name}")
    
    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    # Test
    np.random.seed(42)
    y_true = np.random.rand(200) * 2 + 6  # pH range 6-8
    y_pred = y_true + np.random.randn(200) * 0.1
    
    plot_predictions_vs_actual(y_true, y_pred, model_name='Test', show=False)
    plot_residuals(y_true, y_pred, model_name='Test', show=False)
    
    print("✓ Forecasting plots test passed!")
