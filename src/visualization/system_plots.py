"""
============================================================================
System-Wide Evaluation Plots
Overall system performance and comparison visualizations
============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List

import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import get_logger

logger = get_logger()

plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
sns.set_style('whitegrid')


def plot_system_performance_summary(classification_metrics: Dict,
                                   forecasting_metrics: Dict,
                                   anomaly_metrics: Dict,
                                   save_path: Path = None, show: bool = False):
    """
    Overall system performance summary
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Classification metrics
    ax1 = fig.add_subplot(gs[0, :])
    cls_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    cls_values = [classification_metrics.get(m, 0) for m in cls_metrics]
    cls_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    
    bars1 = ax1.bar(cls_names, cls_values, color='steelblue', alpha=0.8)
    ax1.set_ylabel('Score', fontweight='bold')
    ax1.set_title('Classification Performance', fontweight='bold', fontsize=12)
    ax1.set_ylim([0, 1])
    ax1.grid(axis='y', alpha=0.3)
    
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Forecasting metrics
    ax2 = fig.add_subplot(gs[1, :2])
    fc_metrics = ['rmse', 'mae', 'mape']
    fc_values = [forecasting_metrics.get(m, 0) for m in fc_metrics]
    fc_names = ['RMSE', 'MAE', 'MAPE (%)']
    
    bars2 = ax2.bar(fc_names, fc_values, color='coral', alpha=0.8)
    ax2.set_ylabel('Error', fontweight='bold')
    ax2.set_title('Forecasting Performance', fontweight='bold', fontsize=12)
    ax2.grid(axis='y', alpha=0.3)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    # R² separately (different scale)
    ax3 = fig.add_subplot(gs[1, 2])
    r2_value = forecasting_metrics.get('r2', 0)
    bar3 = ax3.bar(['R²'], [r2_value], color='coral', alpha=0.8)
    ax3.set_ylabel('Score', fontweight='bold')
    ax3.set_title('R² Score', fontweight='bold', fontsize=12)
    ax3.set_ylim([0, 1])
    ax3.grid(axis='y', alpha=0.3)
    ax3.text(0, r2_value, f'{r2_value:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Anomaly detection
    ax4 = fig.add_subplot(gs[2, :])
    if 'precision' in anomaly_metrics:
        anom_metrics = ['precision', 'recall', 'f1_score']
        anom_values = [anomaly_metrics.get(m, 0) for m in anom_metrics]
        anom_names = ['Precision', 'Recall', 'F1-Score']
        
        bars4 = ax4.bar(anom_names, anom_values, color='lightgreen', alpha=0.8)
        ax4.set_ylabel('Score', fontweight='bold')
        ax4.set_title('Anomaly Detection Performance', fontweight='bold', fontsize=12)
        ax4.set_ylim([0, 1])
        ax4.grid(axis='y', alpha=0.3)
        
        for bar in bars4:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    else:
        ax4.text(0.5, 0.5, 'Anomaly metrics not available', 
                transform=ax4.transAxes, ha='center', fontsize=12)
        ax4.axis('off')
    
    plt.suptitle('Complete System Performance Summary', 
                fontweight='bold', fontsize=16)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.success(f"  Saved: {save_path.name}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_alert_statistics(alert_stats: Dict,
                         save_path: Path = None, show: bool = False):
    """
    Alert system statistics visualization
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Alert level distribution
    if 'by_level' in alert_stats:
        levels = ['NORMAL', 'LOW', 'MEDIUM', 'CRITICAL']
        counts = [alert_stats['by_level'].get(l, 0) for l in levels]
        colors = ['green', 'yellow', 'orange', 'red']
        
        axes[0].bar(levels, counts, color=colors, alpha=0.7, edgecolor='black')
        axes[0].set_ylabel('Count', fontweight='bold')
        axes[0].set_xlabel('Alert Level', fontweight='bold')
        axes[0].set_title('Alert Level Distribution', fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)
        
        for i, (level, count) in enumerate(zip(levels, counts)):
            axes[0].text(i, count, str(count), ha='center', va='bottom', fontsize=10)
    
    # Alert sources
    if 'by_source' in alert_stats:
        sources = list(alert_stats['by_source'].keys())
        source_counts = list(alert_stats['by_source'].values())
        
        axes[1].pie(source_counts, labels=sources, autopct='%1.1f%%',
                   startangle=90, colors=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[1].set_title('Alert Sources Distribution', fontweight='bold')
    
    plt.suptitle('Early Warning System Statistics', fontweight='bold', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.success(f"  Saved: {save_path.name}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_computation_time_comparison(timing_data: Dict,
                                    save_path: Path = None, show: bool = False):
    """
    Model training and inference time comparison
    """
    models = list(timing_data.keys())
    times = list(timing_data.values())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.barh(models, times, color='skyblue', alpha=0.8, edgecolor='black')
    ax.set_xlabel('Time (seconds)', fontweight='bold')
    ax.set_ylabel('Model/Component', fontweight='bold')
    ax.set_title('Computation Time Comparison', fontweight='bold', fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    
    for bar, time in zip(bars, times):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
               f'{time:.2f}s', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.success(f"  Saved: {save_path.name}")
    
    if show:
        plt.show()
    else:
        plt.close()


def create_results_table(all_metrics: Dict,
                        save_path: Path = None):
    """
    Generate LaTeX-formatted results table for IEEE paper
    """
    # Classification
    cls_data = []
    for model_name, metrics in all_metrics.get('classification', {}).items():
        cls_data.append({
            'Model': model_name.replace('_', ' ').title(),
            'Accuracy': f"{metrics.get('accuracy', 0):.4f}",
            'Precision': f"{metrics.get('precision', 0):.4f}",
            'Recall': f"{metrics.get('recall', 0):.4f}",
            'F1-Score': f"{metrics.get('f1_score', 0):.4f}",
            'ROC-AUC': f"{metrics.get('roc_auc', 0):.4f}"
        })
    
    # Forecasting
    fc_data = []
    for model_name, metrics in all_metrics.get('forecasting', {}).items():
        fc_data.append({
            'Model': model_name.replace('_', ' ').title(),
            'RMSE': f"{metrics.get('rmse', 0):.6f}",
            'MAE': f"{metrics.get('mae', 0):.6f}",
            'MAPE': f"{metrics.get('mape', 0):.2f}%",
            'R²': f"{metrics.get('r2', 0):.4f}"
        })
    
    if save_path:
        with open(save_path, 'w') as f:
            # Classification table
            f.write("% Classification Results\n")
            f.write("\\begin{table}[h]\n")
            f.write("\\centering\n")
            f.write("\\caption{Classification Model Performance}\n")
            f.write("\\begin{tabular}{lccccc}\n")
            f.write("\\hline\n")
            f.write("Model & Accuracy & Precision & Recall & F1-Score & ROC-AUC \\\\\n")
            f.write("\\hline\n")
            for row in cls_data:
                f.write(f"{row['Model']} & {row['Accuracy']} & {row['Precision']} & {row['Recall']} & {row['F1-Score']} & {row['ROC-AUC']} \\\\\n")
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n\n")
            
            # Forecasting table
            f.write("% Forecasting Results\n")
            f.write("\\begin{table}[h]\n")
            f.write("\\centering\n")
            f.write("\\caption{Forecasting Model Performance}\n")
            f.write("\\begin{tabular}{lcccc}\n")
            f.write("\\hline\n")
            f.write("Model & RMSE & MAE & MAPE & R² \\\\\n")
            f.write("\\hline\n")
            for row in fc_data:
                f.write(f"{row['Model']} & {row['RMSE']} & {row['MAE']} & {row['MAPE']} & {row['R²']} \\\\\n")
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")
        
        logger.success(f"  LaTeX tables saved: {save_path.name}")


if __name__ == "__main__":
    # Test
    cls_m = {'accuracy': 0.85, 'precision': 0.82, 'recall': 0.88, 'f1_score': 0.85, 'roc_auc': 0.90}
    fc_m = {'rmse': 0.025, 'mae': 0.018, 'mape': 2.5, 'r2': 0.92}
    anom_m = {'precision': 0.78, 'recall': 0.85, 'f1_score': 0.81}
    
    plot_system_performance_summary(cls_m, fc_m, anom_m, show=False)
    
    print("✓ System plots test passed!")
