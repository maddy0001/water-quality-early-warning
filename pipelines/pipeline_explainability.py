"""
============================================================================
Explainability Pipeline
Generates comprehensive model interpretability analysis
============================================================================
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any

sys.path.append(str(Path(__file__).parent.parent))

from config import PLOTS_DIR, METRICS_DIR
from utils.logger import get_logger
from utils.file_handler import FileHandler
from data.data_loader import WaterPotabilityLoader
from data.preprocessing_classification import ClassificationPreprocessor
from models.classification.random_forest_model import RandomForestModel
from explainability.shap_explainer import SHAPExplainer, FeatureImportanceAnalyzer
from explainability.visualizations import create_explainability_report

logger = get_logger()


def run_explainability_pipeline(
    use_pretrained: bool = True,
    generate_plots: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Full explainability pipeline:
    1. Load model and data
    2. Compute SHAP values
    3. Analyze feature importance
    4. Generate visualizations
    5. Create interpretability report
    
    Args:
        use_pretrained: Load existing model vs train new
        generate_plots: Create visualization plots
        verbose: Print progress
    
    Returns:
        Complete explainability results
    """
    pipeline_start = time.time()

    if verbose:
        logger.section("EXPLAINABILITY PIPELINE")

    # -------------------------------------------------------------------------
    # STEP 1: Load Data
    # -------------------------------------------------------------------------
    if verbose:
        logger.info("\n[1/5] Loading data...")

    loader = WaterPotabilityLoader()
    data, success = loader.load(verbose=False)
    
    if not success:
        logger.error("Failed to load data!")
        return {'success': False}
    
    X, y = loader.get_features_and_target()
    logger.success(f"  Loaded {len(X)} samples")

    # -------------------------------------------------------------------------
    # STEP 2: Preprocess
    # -------------------------------------------------------------------------
    if verbose:
        logger.info("\n[2/5] Preprocessing...")

    preprocessor = ClassificationPreprocessor()
    X_processed, y_processed = preprocessor.fit_transform(X, y, verbose=False)
    X_train, X_test, y_train, y_test = preprocessor.split_data(
        X_processed, y_processed, verbose=False
    )
    
    feature_names = preprocessor.feature_names
    logger.success(f"  Features: {len(feature_names)}")

    # -------------------------------------------------------------------------
    # STEP 3: Load/Train Model
    # -------------------------------------------------------------------------
    if verbose:
        logger.info("\n[3/5] Loading model...")

    rf_model = RandomForestModel()
    
    if use_pretrained:
        try:
            rf_model.load()
            logger.success("  Loaded pretrained Random Forest")
        except:
            logger.warning("  Pretrained model not found, training new...")
            rf_model.train(X_train, y_train, feature_names=feature_names, verbose=False)
    else:
        rf_model.train(X_train, y_train, feature_names=feature_names, verbose=False)

    # -------------------------------------------------------------------------
    # STEP 4: Compute Explanations
    # -------------------------------------------------------------------------
    if verbose:
        logger.info("\n[4/5] Computing explanations...")

    # Tree-based importance
    importance_analyzer = FeatureImportanceAnalyzer()
    rf_importance = importance_analyzer.analyze_tree_importance(
        rf_model.model, feature_names, 'Random Forest'
    )
    
    # SHAP values
    shap_explainer = SHAPExplainer()
    shap_results = shap_explainer.explain_model(
        rf_model.model, 
        X_test,
        model_type='tree',
        max_samples=100,
        verbose=verbose
    )
    
    shap_importance = shap_explainer.get_feature_importance()
    
    # Individual prediction explanation
    sample_explanation = shap_explainer.explain_prediction(
        sample_idx=0, verbose=verbose
    )

    # -------------------------------------------------------------------------
    # STEP 5: Generate Visualizations
    # -------------------------------------------------------------------------
    if generate_plots and verbose:
        logger.info("\n[5/5] Generating visualizations...")
        
        plots_dir = PLOTS_DIR / 'explainability'
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Comprehensive report
        feature_importance = {
            'rf': rf_importance,
            'shap': shap_importance
        }
        
        # Check if XGBoost available
        try:
            from models.classification.xgboost_model import XGBoostModel
            xgb_model = XGBoostModel()
            xgb_model.load()
            xgb_importance = xgb_model.get_feature_importance()
            feature_importance['xgb'] = xgb_importance
            logger.info("  XGBoost importance included")
        except:
            logger.info("  XGBoost not available, skipping")
        
        create_explainability_report(
            feature_importance=feature_importance,
            X=X_processed,
            y=y_processed,
            output_dir=plots_dir,
            top_n=9
        )
        
        # SHAP plots
        try:
            shap_explainer.plot_summary(
                save_path=plots_dir / 'shap_summary.png'
            )
            
            shap_explainer.plot_waterfall(
                sample_idx=0,
                save_path=plots_dir / 'shap_waterfall_sample0.png'
            )
            
            logger.success("  All visualizations generated!")
        except Exception as e:
            logger.warning(f"  Some SHAP plots failed: {e}")

    # -------------------------------------------------------------------------
    # Save Results
    # -------------------------------------------------------------------------
    if verbose:
        logger.info("\nSaving results...")
    
    # Top features summary
    top_features_summary = {
        'random_forest_top10': rf_importance.head(10).to_dict('records'),
        'shap_top10': shap_importance.head(10).to_dict('records'),
    }
    
    FileHandler.save_json(
        top_features_summary,
        METRICS_DIR / 'feature_importance_summary.json'
    )
    
    total_time = time.time() - pipeline_start

    logger.section("EXPLAINABILITY PIPELINE COMPLETE")
    logger.info(f"  Total time: {total_time:.1f}s")
    logger.info(f"\n  Top 5 Most Important Features:")
    for i, row in rf_importance.head(5).iterrows():
        logger.info(f"    {i+1}. {row['feature']:<25} {row['importance']:.4f}")

    return {
        'success': True,
        'rf_importance': rf_importance,
        'shap_importance': shap_importance,
        'shap_explainer': shap_explainer,
        'sample_explanation': sample_explanation,
        'feature_names': feature_names,
        'total_time': total_time
    }


if __name__ == "__main__":
    results = run_explainability_pipeline(
        use_pretrained=True,
        generate_plots=True,
        verbose=True
    )

    if results['success']:
        print("\n" + "="*70)
        print("✓ Explainability pipeline complete!")
        print(f"  Top feature: {results['rf_importance'].iloc[0]['feature']}")
        print(f"  Total time: {results['total_time']:.1f}s")
        print("="*70)
