"""
============================================================================
Explainability Visualizations
Creates comprehensive plots for model interpretability
============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import PLOTS_DIR
from utils.logger import get_logger

logger = get_logger()

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100


def plot_feature_importance_comparison(
    rf_importance: pd.DataFrame,
    xgb_importance: pd.DataFrame = None,
    shap_importance: pd.DataFrame = None,
    top_n: int = 10,
    save_path: Path = None,
    show: bool = False
):
    """
    Compare feature importance across different methods
    
    Args:
        rf_importance: Random Forest importance
        xgb_importance: XGBoost importance (optional)
        shap_importance: SHAP importance (optional)
        top_n: Number of top features
        save_path: Where to save
        show: Display plot
    """
    logger.info("  Creating feature importance comparison...")
    
    # Determine number of subplots
    n_methods = 1 + (xgb_importance is not None) + (shap_importance is not None)
    
    fig, axes = plt.subplots(1, n_methods, figsize=(6*n_methods, 8))
    if n_methods == 1:
        axes = [axes]
    
    idx = 0
    
    # Random Forest
    rf_top = rf_importance.head(top_n).sort_values('importance')
    axes[idx].barh(range(len(rf_top)), rf_top['importance'], color='forestgreen')
    axes[idx].set_yticks(range(len(rf_top)))
    axes[idx].set_yticklabels(rf_top['feature'])
    axes[idx].set_xlabel('Importance')
    axes[idx].set_title('Random Forest\nFeature Importance', fontweight='bold')
    axes[idx].grid(axis='x', alpha=0.3)
    idx += 1
    
    # XGBoost
    if xgb_importance is not None:
        xgb_top = xgb_importance.head(top_n).sort_values('importance')
        axes[idx].barh(range(len(xgb_top)), xgb_top['importance'], color='coral')
        axes[idx].set_yticks(range(len(xgb_top)))
        axes[idx].set_yticklabels(xgb_top['feature'])
        axes[idx].set_xlabel('Importance')
        axes[idx].set_title('XGBoost\nFeature Importance', fontweight='bold')
        axes[idx].grid(axis='x', alpha=0.3)
        idx += 1
    
    # SHAP
    if shap_importance is not None:
        shap_top = shap_importance.head(top_n).sort_values('importance')
        axes[idx].barh(range(len(shap_top)), shap_top['importance'], color='steelblue')
        axes[idx].set_yticks(range(len(shap_top)))
        axes[idx].set_yticklabels(shap_top['feature'])
        axes[idx].set_xlabel('Mean |SHAP Value|')
        axes[idx].set_title('SHAP\nFeature Importance', fontweight='bold')
        axes[idx].grid(axis='x', alpha=0.3)
    
    plt.suptitle('Feature Importance Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.success(f"  Saved: {save_path.name}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_top_features_heatmap(
    X: pd.DataFrame,
    y: pd.Series,
    top_features: List[str],
    save_path: Path = None,
    show: bool = False
):
    """
    Heatmap showing correlation of top features with target
    """
    logger.info("  Creating feature correlation heatmap...")
    
    # Select top features + target
    data = X[top_features].copy()
    data['Target'] = y
    
    # Compute correlation
    corr = data.corr()
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        vmin=-1, vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Correlation'}
    )
    
    plt.title('Feature Correlation Heatmap\n(Top Features)', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.success(f"  Saved: {save_path.name}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_feature_distributions(
    X: pd.DataFrame,
    y: pd.Series,
    top_features: List[str],
    save_path: Path = None,
    show: bool = False
):
    """
    Distribution plots for top features, split by class
    """
    logger.info("  Creating feature distribution plots...")
    
    n_features = len(top_features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for idx, feature in enumerate(top_features):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        
        # Split by class
        for cls in sorted(y.unique()):
            data = X[y == cls][feature]
            ax.hist(data, alpha=0.6, bins=30, 
                   label=f'Class {cls}', density=True)
        
        ax.set_xlabel(feature)
        ax.set_ylabel('Density')
        ax.set_title(f'{feature} Distribution', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(top_features), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Top Features: Distribution by Class', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.success(f"  Saved: {save_path.name}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_prediction_explanation(
    feature_contributions: pd.DataFrame,
    base_value: float,
    prediction: float,
    sample_id: int = 0,
    save_path: Path = None,
    show: bool = False
):
    """
    Waterfall-style plot showing how features contribute to prediction
    """
    logger.info(f"  Creating prediction explanation for sample {sample_id}...")
    
    # Get top contributors (positive and negative)
    contrib = feature_contributions.copy()
    contrib = contrib.nlargest(10, 'shap_value', keep='all')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['green' if x > 0 else 'red' for x in contrib['shap_value']]
    y_pos = range(len(contrib))
    
    ax.barh(y_pos, contrib['shap_value'], color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(contrib['feature'])
    ax.set_xlabel('SHAP Value (impact on prediction)')
    ax.set_title(f'Prediction Explanation - Sample {sample_id}\n' +
                 f'Base: {base_value:.3f} → Prediction: {prediction:.3f}',
                 fontweight='bold')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.success(f"  Saved: {save_path.name}")
    
    if show:
        plt.show()
    else:
        plt.close()


def create_explainability_report(
    feature_importance: Dict[str, pd.DataFrame],
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: Path,
    top_n: int = 10
):
    """
    Generate complete explainability report with all plots
    
    Args:
        feature_importance: Dict with 'rf', 'xgb', 'shap' importance DataFrames
        X: Feature data
        y: Target
        output_dir: Where to save plots
        top_n: Number of top features
    """
    logger.section("GENERATING EXPLAINABILITY REPORT")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Feature importance comparison
    plot_feature_importance_comparison(
        rf_importance=feature_importance.get('rf'),
        xgb_importance=feature_importance.get('xgb'),
        shap_importance=feature_importance.get('shap'),
        top_n=top_n,
        save_path=output_dir / 'feature_importance_comparison.png'
    )
    
    # 2. Get top features from RF
    top_features = feature_importance['rf'].head(top_n)['feature'].tolist()
    
    # 3. Correlation heatmap
    plot_top_features_heatmap(
        X, y, top_features,
        save_path=output_dir / 'feature_correlation_heatmap.png'
    )
    
    # 4. Distribution plots
    plot_feature_distributions(
        X, y, top_features[:9],  # Top 9 for nice 3x3 grid
        save_path=output_dir / 'feature_distributions.png'
    )
    
    logger.success(f"Explainability report saved to {output_dir}")
    logger.info(f"  Generated 3 visualization plots")


if __name__ == "__main__":
    # Test
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(200, 10), 
                     columns=[f'Feature_{i}' for i in range(10)])
    y = pd.Series((np.random.randn(200) > 0).astype(int))
    
    rf_imp = pd.DataFrame({
        'feature': X.columns,
        'importance': np.random.rand(10)
    }).sort_values('importance', ascending=False)
    
    plot_feature_importance_comparison(rf_imp, top_n=5, show=False)
    print("✓ Visualization test passed!")
