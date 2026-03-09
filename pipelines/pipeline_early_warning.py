"""
============================================================================
Early Warning Pipeline
Complete end-to-end early warning system
============================================================================
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any

sys.path.append(str(Path(__file__).parent.parent))

from config import PLOTS_DIR
from utils.logger import get_logger
from data.data_loader import WaterPotabilityLoader
from data.preprocessing_classification import ClassificationPreprocessor
from early_warning.alert_system import AlertLevel
from early_warning.monitoring_engine import WaterQualityMonitor
from early_warning.alert_logger import AlertLogger, NotificationSystem, generate_dashboard_data

logger = get_logger()


def run_early_warning_pipeline(
    n_samples: int = 100,
    enable_notifications: bool = False,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Complete early warning pipeline:
    1. Load monitoring system
    2. Process water samples
    3. Generate alerts
    4. Log and report
    5. Create dashboard data
    
    Args:
        n_samples: Number of samples to monitor
        enable_notifications: Enable notification system
        verbose: Print progress
    
    Returns:
        Complete results and statistics
    """
    pipeline_start = time.time()

    if verbose:
        logger.section("EARLY WARNING SYSTEM PIPELINE")

    # -------------------------------------------------------------------------
    # STEP 1: Initialize System
    # -------------------------------------------------------------------------
    if verbose:
        logger.info("\n[1/5] Initializing early warning system...")

    monitor = WaterQualityMonitor()
    alert_logger = AlertLogger()
    notifier = NotificationSystem()
    notifier.enabled = enable_notifications

    # Load models
    monitor.load_models(verbose=verbose)

    # -------------------------------------------------------------------------
    # STEP 2: Load Test Data
    # -------------------------------------------------------------------------
    if verbose:
        logger.info("\n[2/5] Loading water quality data...")

    loader = WaterPotabilityLoader()
    data, success = loader.load(verbose=False)
    
    if not success:
        logger.error("Failed to load data!")
        return {'success': False}
    
    X, y = loader.get_features_and_target()
    
    # Preprocess
    preprocessor = ClassificationPreprocessor()
    X_processed, _ = preprocessor.fit_transform(X, y, verbose=False)
    
    # Take sample
    X_sample = X_processed.head(n_samples)
    
    logger.success(f"  Loaded {len(X_sample)} samples for monitoring")

    # -------------------------------------------------------------------------
    # STEP 3: Run Monitoring
    # -------------------------------------------------------------------------
    if verbose:
        logger.info("\n[3/5] Running real-time monitoring...")

    alerts = monitor.monitor_batch(
        X_sample,
        feature_names=preprocessor.feature_names,
        locations=[f"Site_{i}" for i in range(len(X_sample))],
        verbose=verbose
    )

    # -------------------------------------------------------------------------
    # STEP 4: Process Alerts
    # -------------------------------------------------------------------------
    if verbose:
        logger.info("\n[4/5] Processing and logging alerts...")

    # Log all alerts
    alert_logger.log_batch(alerts)
    alert_logger.flush()

    # Send notifications for critical alerts
    critical_count = 0
    for alert in alerts:
        if notifier.notify_if_critical(alert):
            critical_count += 1
    
    if verbose and critical_count > 0:
        logger.warning(f"  Sent {critical_count} critical notifications")

    # -------------------------------------------------------------------------
    # STEP 5: Generate Reports
    # -------------------------------------------------------------------------
    if verbose:
        logger.info("\n[5/5] Generating reports...")

    # Get statistics
    stats = monitor.get_alert_statistics()
    
    # Generate dashboard data
    dashboard = generate_dashboard_data(alert_logger, hours=24)
    
    # Detailed report
    report = alert_logger.generate_report(hours=24)

    # -------------------------------------------------------------------------
    # Print Summary
    # -------------------------------------------------------------------------
    if verbose:
        _print_summary(stats, dashboard, report)

    total_time = time.time() - pipeline_start

    logger.section("EARLY WARNING PIPELINE COMPLETE")
    logger.info(f"  Total time: {total_time:.1f}s")
    logger.info(f"  Samples monitored: {len(alerts)}")
    logger.info(f"  Critical alerts: {stats.get('critical_count', 0)}")

    return {
        'success': True,
        'alerts': alerts,
        'statistics': stats,
        'dashboard': dashboard,
        'report': report,
        'monitor': monitor,
        'alert_logger': alert_logger,
        'total_time': total_time
    }


def _print_summary(stats: Dict, dashboard: Dict, report: Dict):
    """Print formatted summary"""
    logger.section("EARLY WARNING SUMMARY")
    
    # Overall stats
    logger.subsection("Overall Statistics")
    logger.info(f"  Total alerts: {stats.get('total_alerts', 0)}")
    logger.info(f"  Risk score: {dashboard['summary']['risk_score']}/100")
    logger.info(f"  Alert rate: {dashboard['summary']['alert_rate_per_hour']:.2f}/hour")
    
    # By level
    if 'by_level' in stats:
        logger.subsection("Alerts by Level")
        for level in ['NORMAL', 'LOW', 'MEDIUM', 'CRITICAL']:
            count = stats['by_level'].get(level, 0)
            if count > 0:
                emoji = {'NORMAL': '✓', 'LOW': '⚠️', 'MEDIUM': '⚠️⚠️', 'CRITICAL': '🚨'}
                logger.info(f"  {emoji.get(level, '')} {level:<10} {count:>4}")
    
    # By source
    if 'by_source' in stats:
        logger.subsection("Alerts by Source")
        for source, count in stats['by_source'].items():
            logger.info(f"  {source:<20} {count:>4}")
    
    # Critical alerts
    if stats.get('critical_count', 0) > 0:
        logger.subsection("⚠️  CRITICAL ALERTS DETECTED")
        logger.warning(f"  {stats['critical_count']} samples require immediate investigation!")
        
        if report.get('critical_alerts'):
            logger.info("\n  Most recent critical alerts:")
            for alert in report['critical_alerts'][-3:]:
                location = alert.get('location', 'Unknown')
                source = alert.get('source', 'Unknown')
                logger.info(f"    • {location} ({source})")


def simulate_real_time_monitoring(duration_seconds: int = 10,
                                  verbose: bool = True):
    """
    Simulate real-time monitoring (for demo purposes)
    
    Args:
        duration_seconds: How long to simulate
        verbose: Print updates
    """
    import numpy as np
    from time import sleep
    
    logger.section("REAL-TIME MONITORING SIMULATION")
    logger.info(f"  Simulating {duration_seconds} seconds of monitoring...")
    
    monitor = WaterQualityMonitor()
    monitor.load_models(verbose=False)
    
    alert_logger = AlertLogger()
    
    for t in range(duration_seconds):
        # Generate random sample
        sample = np.random.randn(9)
        
        # Check sample
        alert = monitor.check_sample(sample, location=f"Sensor_{t%5}", verbose=False)
        alert_logger.log_alert(alert)
        
        if verbose and alert.level != AlertLevel.NORMAL:
            logger.warning(f"  t={t}s: {alert.level.name} at {alert.location}")
        
        sleep(1)
    
    alert_logger.flush()
    
    # Summary
    stats = monitor.get_alert_statistics()
    logger.success(f"\n  Monitoring complete: {stats['total_alerts']} alerts generated")
    logger.info(f"  Distribution: {stats.get('by_level', {})}")


if __name__ == "__main__":
    # Run pipeline
    results = run_early_warning_pipeline(
        n_samples=100,
        enable_notifications=True,
        verbose=True
    )

    if results['success']:
        print("\n" + "="*70)
        print("✓ Early warning pipeline complete!")
        print(f"  Risk score: {results['dashboard']['summary']['risk_score']}/100")
        print(f"  Critical alerts: {results['statistics'].get('critical_count', 0)}")
        print(f"  Total time: {results['total_time']:.1f}s")
        print("="*70)
        
        # Optional: Simulate real-time
        # print("\n" + "="*70)
        # simulate_real_time_monitoring(duration_seconds=10, verbose=True)
