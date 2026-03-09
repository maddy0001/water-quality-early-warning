"""
================================================================================
        WATER CONTAMINATION EARLY WARNING SYSTEM - MAIN ENTRY POINT
================================================================================

AI-Powered Water Quality Monitoring and Contamination Detection System

Components:
  • Classification Models (RF, XGBoost, SVM, Ensemble)
  • Time Series Forecasting (BiLSTM + GRU with Attention)
  • Anomaly Detection (Isolation Forest, LOF, OCSVM, Autoencoder)
  • Explainability (SHAP)
  • Early Warning System (4-level alerts)
  • Comprehensive Evaluation & Visualization

Author: M.Tech Student
IEEE Paper Submission: 2026
================================================================================
"""

import sys
import argparse
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from config import RANDOM_SEED
from utils.logger import get_logger
from utils.helpers import set_random_seeds

logger = get_logger()


def print_banner():
    """Print system banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                                                                      ║
    ║        WATER CONTAMINATION EARLY WARNING SYSTEM (AI-POWERED)        ║
    ║                                                                      ║
    ║  Components: Classification + Forecasting + Anomaly + Explainability║
    ║  Models: RF, XGBoost, SVM, BiLSTM, GRU, IF, LOF, OCSVM, Autoencoder║
    ║  Alerts: 4-Level System (NORMAL → LOW → MEDIUM → CRITICAL)         ║
    ║                                                                      ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def run_full_system(config: dict):
    """
    Run complete system workflow
    
    Workflow:
    1. Data preprocessing
    2. Train all models (Classification + Forecasting + Anomaly)
    3. Generate explainability analysis
    4. Run early warning system
    5. Comprehensive evaluation
    6. Generate all visualizations
    """
    logger.section("RUNNING COMPLETE SYSTEM")
    
    # Set seeds
    set_random_seeds(RANDOM_SEED)
    
    total_start = time.time()
    
    # Step 1: Classification Pipeline
    if config.get('run_classification', True):
        logger.section("STEP 1/5: CLASSIFICATION")
        from pipelines.pipeline_classification import run_classification_pipeline
        
        cls_results = run_classification_pipeline(
            save_models=config.get('save_models', True),
            verbose=config.get('verbose', True)
        )
        logger.success("✓ Classification complete")
    
    # Step 2: Forecasting Pipeline
    if config.get('run_forecasting', True):
        logger.section("STEP 2/5: FORECASTING")
        from pipelines.pipeline_forecasting import run_forecasting_pipeline
        
        fc_results = run_forecasting_pipeline(
            save_models=config.get('save_models', True),
            verbose=config.get('verbose', True)
        )
        logger.success("✓ Forecasting complete")
    
    # Step 3: Anomaly Detection Pipeline
    if config.get('run_anomaly', True):
        logger.section("STEP 3/5: ANOMALY DETECTION")
        from pipelines.pipeline_anomaly import run_anomaly_pipeline
        
        anom_results = run_anomaly_pipeline(
            save_models=config.get('save_models', True),
            use_autoencoder=config.get('use_autoencoder', True),
            verbose=config.get('verbose', True)
        )
        logger.success("✓ Anomaly detection complete")
    
    # Step 4: Explainability Pipeline
    if config.get('run_explainability', True):
        logger.section("STEP 4/5: EXPLAINABILITY")
        from pipelines.pipeline_explainability import run_explainability_pipeline
        
        exp_results = run_explainability_pipeline(
            use_pretrained=True,
            generate_plots=config.get('generate_plots', True),
            verbose=config.get('verbose', True)
        )
        logger.success("✓ Explainability complete")
    
    # Step 5: Evaluation & Visualization
    if config.get('run_evaluation', True):
        logger.section("STEP 5/5: EVALUATION & VISUALIZATION")
        from pipelines.pipeline_evaluation import run_evaluation_pipeline
        
        eval_results = run_evaluation_pipeline(
            verbose=config.get('verbose', True)
        )
        logger.success("✓ Evaluation complete")
    
    total_time = time.time() - total_start
    
    # Final Summary
    logger.section("COMPLETE SYSTEM EXECUTION SUMMARY")
    logger.info(f"  Total execution time: {total_time:.1f}s ({total_time/60:.1f} min)")
    logger.info(f"  All pipelines completed successfully!")
    logger.info(f"\n  📊 Results saved to: outputs/")
    logger.info(f"  📈 Plots saved to: outputs/plots/")
    logger.info(f"  🤖 Models saved to: models/")
    
    return {
        'success': True,
        'total_time': total_time
    }


def run_early_warning_demo(n_samples: int = 50):
    """
    Run early warning system demo
    
    Monitors water quality samples in real-time and generates alerts
    """
    logger.section("EARLY WARNING SYSTEM DEMO")
    
    from pipelines.pipeline_early_warning import run_early_warning_pipeline
    
    results = run_early_warning_pipeline(
        n_samples=n_samples,
        enable_notifications=True,
        verbose=True
    )
    
    if results['success']:
        logger.section("DEMO COMPLETE")
        logger.info(f"  Samples monitored: {n_samples}")
        logger.info(f"  Alerts generated: {results['statistics']['total_alerts']}")
        logger.info(f"  Critical alerts: {results['statistics'].get('critical_count', 0)}")
        logger.info(f"  Risk score: {results['dashboard']['summary']['risk_score']}/100")
    
    return results


def run_evaluation_only():
    """Run only evaluation and visualization"""
    logger.section("EVALUATION & VISUALIZATION ONLY")
    
    from pipelines.pipeline_evaluation import run_evaluation_pipeline
    
    results = run_evaluation_pipeline(verbose=True)
    
    if results['success']:
        logger.success(f"\n✓ Generated {results['plot_count']} plots")
        logger.info(f"  Saved to: outputs/plots/evaluation/")
    
    return results


def run_train_only():
    """Train all models without evaluation"""
    logger.section("TRAINING ALL MODELS")
    
    # Classification
    logger.info("\n[1/3] Training classification models...")
    from pipelines.pipeline_classification import run_classification_pipeline
    run_classification_pipeline(save_models=True, verbose=True)
    
    # Forecasting
    logger.info("\n[2/3] Training forecasting models...")
    from pipelines.pipeline_forecasting import run_forecasting_pipeline
    run_forecasting_pipeline(save_models=True, verbose=True)
    
    # Anomaly
    logger.info("\n[3/3] Training anomaly detectors...")
    from pipelines.pipeline_anomaly import run_anomaly_pipeline
    run_anomaly_pipeline(save_models=True, verbose=True)
    
    logger.success("\n✓ All models trained and saved!")


def print_system_info():
    """Print system information"""
    import tensorflow as tf
    import numpy as np
    import pandas as pd
    import sklearn
    
    logger.section("SYSTEM INFORMATION")
    
    logger.info("  Python Packages:")
    logger.info(f"    NumPy:        {np.__version__}")
    logger.info(f"    Pandas:       {pd.__version__}")
    logger.info(f"    Scikit-learn: {sklearn.__version__}")
    logger.info(f"    TensorFlow:   {tf.__version__}")
    
    try:
        import xgboost as xgb
        logger.info(f"    XGBoost:      {xgb.__version__}")
    except:
        logger.warning("    XGBoost:      Not installed")
    
    try:
        import shap
        logger.info(f"    SHAP:         {shap.__version__}")
    except:
        logger.warning("    SHAP:         Not installed")
    
    # GPU Check
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logger.info(f"\n  GPU: {len(gpus)} device(s) available")
        for gpu in gpus:
            logger.info(f"    {gpu.name}")
    else:
        logger.info("\n  GPU: No GPU detected (using CPU)")
    
    logger.info(f"\n  Random Seed: {RANDOM_SEED}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Water Contamination Early Warning System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --full              # Run complete system
  python main.py --train             # Train all models
  python main.py --evaluate          # Generate evaluation plots
  python main.py --demo              # Run early warning demo
  python main.py --info              # Show system info
  python main.py --full --no-plots   # Run without generating plots
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--full', action='store_true',
                           help='Run complete system (all pipelines)')
    mode_group.add_argument('--train', action='store_true',
                           help='Train all models only')
    mode_group.add_argument('--evaluate', action='store_true',
                           help='Run evaluation and generate plots only')
    mode_group.add_argument('--demo', action='store_true',
                           help='Run early warning system demo')
    mode_group.add_argument('--info', action='store_true',
                           help='Show system information')
    
    # Options
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip plot generation')
    parser.add_argument('--no-autoencoder', action='store_true',
                       help='Skip autoencoder (no TensorFlow required)')
    parser.add_argument('--samples', type=int, default=50,
                       help='Number of samples for demo mode')
    parser.add_argument('--quiet', action='store_true',
                       help='Minimal output')
    
    args = parser.parse_args()
    
    # Print banner
    if not args.quiet:
        print_banner()
    
    # Show info
    if args.info:
        print_system_info()
        return
    
    # Configuration
    config = {
        'run_classification': True,
        'run_forecasting': True,
        'run_anomaly': True,
        'run_explainability': True,
        'run_evaluation': True,
        'save_models': True,
        'generate_plots': not args.no_plots,
        'use_autoencoder': not args.no_autoencoder,
        'verbose': not args.quiet
    }
    
    # Execute based on mode
    
    
    if args.full:
        run_full_system(config)
    
    elif args.train:
        run_train_only()
    
    elif args.evaluate:
        run_evaluation_only()
    
    elif args.demo:
        run_early_warning_demo(n_samples=args.samples)
    
    else:
        # Default: show help
        parser.print_help()
        print("\n💡 TIP: Use --full to run the complete system")
        print("        Use --demo for a quick demo")
        print("        Use --info to see system information")


if __name__ == "__main__":
    main()
