"""
============================================================================
SHAP Explainer for Model Interpretability
Explains predictions using Shapley values
============================================================================
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import PLOTS_DIR
from utils.logger import get_logger

logger = get_logger()


class SHAPExplainer:
    """
    SHAP (SHapley Additive exPlanations) for model interpretability
    
    Explains:
    - Global feature importance
    - Individual prediction explanations
    - Feature interactions
    """

    def __init__(self):
        """Initialize SHAP explainer"""
        self.explainer = None
        self.shap_values = None
        self.feature_names = None
        self.base_value = None
        
        try:
            import shap
            self.shap = shap
        except ImportError:
            raise ImportError("SHAP not installed. Run: pip install shap")

    def explain_model(self, model, X: pd.DataFrame, 
                     model_type: str = 'tree',
                     max_samples: int = 100,
                     verbose: bool = True) -> Dict:
        """
        Create SHAP explanations for a model
        
        Args:
            model: Trained model (RF, XGBoost, etc.)
            X: Dataset to explain
            model_type: 'tree', 'linear', or 'kernel'
            max_samples: Max samples for explanation (for speed)
            verbose: Print progress
        
        Returns:
            SHAP values and statistics
        """
        if verbose:
            logger.subsection("Computing SHAP Values")
            logger.info(f"  Model type: {model_type}")
            logger.info(f"  Samples: {min(len(X), max_samples)}")

        self.feature_names = list(X.columns) if hasattr(X, 'columns') else None
        
        # Limit samples for speed
        X_sample = X.sample(min(len(X), max_samples), random_state=42) if len(X) > max_samples else X
        
        # Create explainer based on model type
        if model_type == 'tree':
            self.explainer = self.shap.TreeExplainer(model)
        elif model_type == 'linear':
            self.explainer = self.shap.LinearExplainer(model, X_sample)
        else:  # kernel
            self.explainer = self.shap.KernelExplainer(model.predict, X_sample)
        
        # Compute SHAP values
        self.shap_values = self.explainer.shap_values(X_sample)
        
        # Handle multi-output (classification) - take positive class
        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[1]  # Positive class
        
        self.base_value = self.explainer.expected_value
        if isinstance(self.base_value, (list, np.ndarray)):
            self.base_value = self.base_value[1]
        
        if verbose:
            logger.success(f"  SHAP values computed!")
            logger.info(f"  Shape: {self.shap_values.shape}")
        
        return {
            'shap_values': self.shap_values,
            'base_value': self.base_value,
            'feature_names': self.feature_names
        }

    def get_feature_importance(self, top_n: int = None) -> pd.DataFrame:
        """
        Get global feature importance from SHAP values
        
        Args:
            top_n: Return only top N features
        
        Returns:
            DataFrame with feature importance
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Call explain_model() first.")
        
        # Mean absolute SHAP value per feature
        importance = np.abs(self.shap_values).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names or [f'Feature_{i}' for i in range(len(importance))],
            'importance': importance
        }).sort_values('importance', ascending=False).reset_index(drop=True)
        
        if top_n:
            importance_df = importance_df.head(top_n)
        
        return importance_df

    def plot_summary(self, save_path: Path = None, show: bool = False):
        """
        Create SHAP summary plot (beeswarm)
        Shows feature importance and impact direction
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed.")
        
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 8))
        self.shap.summary_plot(
            self.shap_values,
            features=pd.DataFrame(
                np.zeros_like(self.shap_values),
                columns=self.feature_names
            ),
            feature_names=self.feature_names,
            plot_type="bar",
            show=False
        )
        
        if save_path:
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.success(f"  Summary plot saved: {save_path.name}")
        
        if show:
            plt.show()
        else:
            plt.close()

    def plot_waterfall(self, sample_idx: int = 0, 
                       save_path: Path = None, show: bool = False):
        """
        Waterfall plot for single prediction explanation
        Shows how features contribute to the prediction
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed.")
        
        # Create explanation object
        explanation = self.shap.Explanation(
            values=self.shap_values[sample_idx],
            base_values=self.base_value,
            feature_names=self.feature_names
        )
        
        self.shap.waterfall_plot(explanation, show=False)
        
        if save_path:
            import matplotlib.pyplot as plt
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.success(f"  Waterfall plot saved: {save_path.name}")
        
        if show:
            import matplotlib.pyplot as plt
            plt.show()
        else:
            import matplotlib.pyplot as plt
            plt.close()

    def explain_prediction(self, sample_idx: int = 0, 
                          verbose: bool = True) -> Dict:
        """
        Explain a single prediction
        
        Returns:
            Dictionary with feature contributions
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed.")
        
        shap_vals = self.shap_values[sample_idx]
        
        # Create contribution DataFrame
        contrib_df = pd.DataFrame({
            'feature': self.feature_names,
            'shap_value': shap_vals
        }).sort_values('shap_value', key=abs, ascending=False)
        
        if verbose:
            logger.subsection(f"Prediction Explanation (Sample {sample_idx})")
            logger.info(f"  Base value: {self.base_value:.4f}")
            logger.info(f"  Prediction: {self.base_value + shap_vals.sum():.4f}")
            logger.info("\n  Top contributing features:")
            for _, row in contrib_df.head(5).iterrows():
                direction = "↑" if row['shap_value'] > 0 else "↓"
                logger.info(f"    {row['feature']:<25} {row['shap_value']:>8.4f} {direction}")
        
        return {
            'base_value': float(self.base_value),
            'prediction': float(self.base_value + shap_vals.sum()),
            'contributions': contrib_df
        }


class FeatureImportanceAnalyzer:
    """
    Unified feature importance analysis across models
    Combines tree-based, permutation, and SHAP importance
    """

    def __init__(self):
        self.importance_results = {}

    def analyze_tree_importance(self, model, feature_names: List[str],
                                model_name: str = 'model') -> pd.DataFrame:
        """Get feature importance from tree-based models"""
        logger.info(f"  Computing tree importance for {model_name}...")
        
        importance = model.feature_importances_
        
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False).reset_index(drop=True)
        
        self.importance_results[f'{model_name}_tree'] = df
        return df

    def analyze_permutation_importance(self, model, X, y, 
                                      feature_names: List[str],
                                      model_name: str = 'model',
                                      n_repeats: int = 10) -> pd.DataFrame:
        """Compute permutation importance"""
        from sklearn.inspection import permutation_importance
        
        logger.info(f"  Computing permutation importance for {model_name}...")
        
        result = permutation_importance(
            model, X, y, n_repeats=n_repeats,
            random_state=42, n_jobs=-1
        )
        
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': result.importances_mean,
            'std': result.importances_std
        }).sort_values('importance', ascending=False).reset_index(drop=True)
        
        self.importance_results[f'{model_name}_permutation'] = df
        return df

    def plot_comparison(self, model_name: str, 
                       save_path: Path = None, show: bool = False):
        """Compare different importance methods"""
        import matplotlib.pyplot as plt
        
        tree_key = f'{model_name}_tree'
        perm_key = f'{model_name}_permutation'
        
        if tree_key not in self.importance_results:
            logger.warning(f"  No tree importance for {model_name}")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Tree importance
        df_tree = self.importance_results[tree_key].head(10)
        axes[0].barh(range(len(df_tree)), df_tree['importance'])
        axes[0].set_yticks(range(len(df_tree)))
        axes[0].set_yticklabels(df_tree['feature'])
        axes[0].set_xlabel('Importance')
        axes[0].set_title(f'{model_name} - Tree-based Importance')
        axes[0].invert_yaxis()
        
        # Permutation importance (if available)
        if perm_key in self.importance_results:
            df_perm = self.importance_results[perm_key].head(10)
            axes[1].barh(range(len(df_perm)), df_perm['importance'])
            axes[1].set_yticks(range(len(df_perm)))
            axes[1].set_yticklabels(df_perm['feature'])
            axes[1].set_xlabel('Importance')
            axes[1].set_title(f'{model_name} - Permutation Importance')
            axes[1].invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.success(f"  Comparison plot saved: {save_path.name}")
        
        if show:
            plt.show()
        else:
            plt.close()


if __name__ == "__main__":
    # Test
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=200, n_features=10, random_state=42)
    X_df = pd.DataFrame(X, columns=[f'Feature_{i}' for i in range(10)])
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_df, y)
    
    explainer = SHAPExplainer()
    explainer.explain_model(model, X_df, model_type='tree', verbose=True)
    importance = explainer.get_feature_importance(top_n=5)
    print("\nTop 5 features:")
    print(importance)
    print("\n✓ SHAP explainer test passed!")
