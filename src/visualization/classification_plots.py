"""
============================================================================
Classification Evaluation Plots
Publication-quality visualizations for classification models
============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    classification_report
)

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import PLOTS_DIR
from utils.logger import get_logger

logger = get_logger()

# Publication settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
sns.set_style('whitegrid')


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          model_name: str = 'Model',
                          save_path: Path = None, show: bool = False):
    """
    Confusion matrix heatmap
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Potable', 'Potable'],
                yticklabels=['Non-Potable', 'Potable'],
                ax=ax, cbar_kws={'label': 'Count'})
    
    ax.set_xlabel('Predicted Label', fontweight='bold')
    ax.set_ylabel('True Label', fontweight='bold')
    ax.set_title(f'Confusion Matrix - {model_name}', fontweight='bold', fontsize=12)
    
    # Add accuracy
    acc = np.trace(cm) / np.sum(cm)
    plt.text(0.5, -0.15, f'Accuracy: {acc:.3f}', 
             transform=ax.transAxes, ha='center', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.success(f"  Saved: {save_path.name}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_roc_curves(models_results: Dict[str, Dict],
                   save_path: Path = None, show: bool = False):
    """
    ROC curves for multiple models
    
    Args:
        models_results: Dict[model_name] -> {'y_true': ..., 'y_prob': ...}
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for idx, (name, results) in enumerate(models_results.items()):
        y_true = results['y_true']
        y_prob = results['y_prob']
        
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color=colors[idx % len(colors)], lw=2,
                label=f'{name} (AUC = {roc_auc:.3f})')
    
    # Diagonal line
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random (AUC = 0.500)')
    
    ax.set_xlabel('False Positive Rate', fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontweight='bold')
    ax.set_title('ROC Curves - Model Comparison', fontweight='bold', fontsize=12)
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.success(f"  Saved: {save_path.name}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_precision_recall_curves(models_results: Dict[str, Dict],
                                 save_path: Path = None, show: bool = False):
    """
    Precision-Recall curves for multiple models
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for idx, (name, results) in enumerate(models_results.items()):
        y_true = results['y_true']
        y_prob = results['y_prob']
        
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)
        
        ax.plot(recall, precision, color=colors[idx % len(colors)], lw=2,
                label=f'{name} (AUC = {pr_auc:.3f})')
    
    ax.set_xlabel('Recall', fontweight='bold')
    ax.set_ylabel('Precision', fontweight='bold')
    ax.set_title('Precision-Recall Curves', fontweight='bold', fontsize=12)
    ax.legend(loc='lower left', frameon=True, fancybox=True, shadow=True)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.success(f"  Saved: {save_path.name}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_metrics_comparison(models_metrics: Dict[str, Dict],
                           save_path: Path = None, show: bool = False):
    """
    Bar chart comparing model metrics
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    
    model_names = list(models_metrics.keys())
    n_models = len(model_names)
    n_metrics = len(metrics)
    
    # Create data matrix
    data = np.zeros((n_models, n_metrics))
    for i, model in enumerate(model_names):
        for j, metric in enumerate(metrics):
            data[i, j] = models_metrics[model].get(metric, 0)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(n_metrics)
    width = 0.8 / n_models
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, model in enumerate(model_names):
        offset = (i - n_models/2 + 0.5) * width
        bars = ax.bar(x + offset, data[i], width, 
                     label=model, color=colors[i % len(colors)], alpha=0.8)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Metrics', fontweight='bold')
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Model Performance Comparison', fontweight='bold', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.set_ylim([0, 1.05])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.success(f"  Saved: {save_path.name}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_feature_importance_comparison(importance_dfs: Dict[str, pd.DataFrame],
                                       top_n: int = 10,
                                       save_path: Path = None, show: bool = False):
    """
    Side-by-side feature importance from different models
    """
    n_models = len(importance_dfs)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 8))
    
    if n_models == 1:
        axes = [axes]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for idx, (name, df) in enumerate(importance_dfs.items()):
        ax = axes[idx]
        
        top_features = df.head(top_n).sort_values('importance')
        
        ax.barh(range(len(top_features)), top_features['importance'], 
               color=colors[idx % len(colors)], alpha=0.8)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('Importance', fontweight='bold')
        ax.set_title(f'{name}\nTop {top_n} Features', fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
    
    plt.suptitle('Feature Importance Comparison', fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.success(f"  Saved: {save_path.name}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_classification_report_heatmap(y_true: np.ndarray, y_pred: np.ndarray,
                                       model_name: str = 'Model',
                                       save_path: Path = None, show: bool = False):
    """
    Heatmap of classification report metrics
    """
    from sklearn.metrics import precision_recall_fscore_support
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    data = np.array([precision, recall, f1]).T
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    sns.heatmap(data, annot=True, fmt='.3f', cmap='YlGnBu',
                xticklabels=['Precision', 'Recall', 'F1-Score'],
                yticklabels=['Non-Potable', 'Potable'],
                ax=ax, cbar_kws={'label': 'Score'}, vmin=0, vmax=1)
    
    ax.set_title(f'Classification Metrics - {model_name}', fontweight='bold', fontsize=12)
    
    # Add support counts
    for i, sup in enumerate(support):
        ax.text(3.2, i+0.5, f'n={int(sup)}', va='center', fontsize=9)
    
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
    y_true = np.random.randint(0, 2, 200)
    y_pred = np.random.randint(0, 2, 200)
    y_prob = np.random.rand(200)
    
    plot_confusion_matrix(y_true, y_pred, model_name='Test Model', show=False)
    
    models = {
        'Model A': {'y_true': y_true, 'y_prob': y_prob},
        'Model B': {'y_true': y_true, 'y_prob': np.random.rand(200)}
    }
    plot_roc_curves(models, show=False)
    
    print("✓ Classification plots test passed!")
