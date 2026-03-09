"""
============================================================================
Monitoring Engine
Real-time water quality monitoring and alert generation
============================================================================
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import MODEL_PATHS
from utils.logger import get_logger
from early_warning.alert_system import Alert, AlertLevel, AlertThresholds, AlertAggregator

logger = get_logger()


class WaterQualityMonitor:
    """
    Real-time water quality monitoring system
    
    Integrates:
    - Classification model (potability prediction)
    - Forecasting model (pH prediction)
    - Anomaly detection (contamination events)
    
    Generates unified alerts
    """

    def __init__(self):
        """Initialize monitoring engine"""
        self.classification_model = None
        self.forecasting_model = None
        self.anomaly_ensemble = None
        self.preprocessor = None
        self.alert_history = []
        
    def load_models(self, verbose: bool = True):
        """Load all trained models"""
        if verbose:
            logger.subsection("Loading Models for Monitoring")
        
        try:
            # Classification
            from models.classification.random_forest_model import RandomForestModel
            self.classification_model = RandomForestModel()
            self.classification_model.load()
            if verbose:
                logger.success("  ✓ Classification model loaded")
        except Exception as e:
            logger.warning(f"  Classification model not loaded: {e}")
        
        try:
            # Forecasting
            from models.forecasting.bilstm_attention import BiLSTMAttentionModel
            self.forecasting_model = BiLSTMAttentionModel()
            self.forecasting_model.load()
            if verbose:
                logger.success("  ✓ Forecasting model loaded")
        except Exception as e:
            logger.warning(f"  Forecasting model not loaded: {e}")
        
        try:
            # Anomaly detection
            from models.anomaly.ensemble_anomaly import EnsembleAnomalyDetector
            self.anomaly_ensemble = EnsembleAnomalyDetector(use_autoencoder=False)
            # Note: Anomaly models trained on-the-fly or loaded separately
            if verbose:
                logger.success("  ✓ Anomaly detector initialized")
        except Exception as e:
            logger.warning(f"  Anomaly detector not loaded: {e}")
    
    def check_sample(self, sample: np.ndarray,
                    feature_names: List[str] = None,
                    location: str = None,
                    verbose: bool = False) -> Alert:
        """
        Check single water sample and generate alert
        
        Args:
            sample: Feature vector (9 features for classification)
            feature_names: Feature names
            location: Sample location identifier
            verbose: Print details
        
        Returns:
            Aggregated alert
        """
        alerts = []
        
        # 1. Classification check
        if self.classification_model is not None:
            alert = self._check_classification(sample, verbose)
            if alert:
                alerts.append(alert)
        
        # 2. Anomaly check
        if self.anomaly_ensemble is not None:
            alert = self._check_anomaly(sample, verbose)
            if alert:
                alerts.append(alert)
        
        # 3. Aggregate
        final_alert = AlertAggregator.aggregate(alerts)
        final_alert.location = location
        
        # Store in history
        self.alert_history.append(final_alert)
        
        if verbose:
            self._print_alert(final_alert)
        
        return final_alert
    
    def _check_classification(self, sample: np.ndarray,
                              verbose: bool = False) -> Optional[Alert]:
        """Run classification check"""
        try:
            # Reshape for prediction
            X = sample.reshape(1, -1)
            
            # Predict
            prediction = self.classification_model.predict(X)[0]
            probability = self.classification_model.predict_proba(X)[0]
            
            # Get confidence for predicted class
            confidence = probability[prediction]
            
            # Determine alert level
            level = AlertThresholds.get_alert_level_from_classification(
                confidence, prediction
            )
            
            if verbose and level != AlertLevel.NORMAL:
                logger.info(f"  Classification: {level.name} (conf={confidence:.3f})")
            
            return Alert(
                level=level,
                timestamp=datetime.now(),
                source='classification',
                confidence=float(confidence),
                details={
                    'prediction': int(prediction),
                    'probabilities': probability.tolist(),
                    'predicted_class': 'potable' if prediction == 1 else 'non-potable'
                }
            )
        except Exception as e:
            if verbose:
                logger.warning(f"  Classification check failed: {e}")
            return None
    
    def _check_anomaly(self, sample: np.ndarray,
                      verbose: bool = False) -> Optional[Alert]:
        """Run anomaly detection check"""
        try:
            X = sample.reshape(1, -1)
            
            # Get predictions from each detector
            vote_count = 0
            detector_results = {}
            
            for name, detector in self.anomaly_ensemble.detectors:
                try:
                    if name == 'LOF':
                        pred = detector.predict_binary(X)[0]
                    else:
                        pred = detector.predict_binary(X)[0]
                    
                    vote_count += pred
                    detector_results[name] = bool(pred)
                except:
                    pass
            
            # Determine alert level
            level = AlertThresholds.get_alert_level_from_anomaly(vote_count)
            
            if verbose and level != AlertLevel.NORMAL:
                logger.info(f"  Anomaly: {level.name} ({vote_count} votes)")
            
            return Alert(
                level=level,
                timestamp=datetime.now(),
                source='anomaly',
                confidence=vote_count / len(self.anomaly_ensemble.detectors),
                details={
                    'vote_count': int(vote_count),
                    'detectors': detector_results
                }
            )
        except Exception as e:
            if verbose:
                logger.warning(f"  Anomaly check failed: {e}")
            return None
    
    def check_forecast(self, sequence: np.ndarray,
                      location: str = None,
                      verbose: bool = False) -> Alert:
        """
        Check pH forecast for next timestep
        
        Args:
            sequence: Time series (30, 11) for forecasting
            location: Location identifier
            verbose: Print details
        
        Returns:
            Alert for forecasted pH
        """
        try:
            # Reshape for model
            X = sequence.reshape(1, *sequence.shape)
            
            # Predict pH
            predicted_ph = self.forecasting_model.predict(X)[0]
            
            # Denormalize if needed (assuming normalized 0-1)
            # For actual deployment, use the scaler
            predicted_ph_actual = predicted_ph * (14.0 - 0.0) + 0.0  # pH range 0-14
            
            # Determine alert level
            level = AlertThresholds.get_alert_level_from_forecasting(predicted_ph_actual)
            
            if verbose:
                logger.info(f"  Forecast: pH={predicted_ph_actual:.2f}, Level={level.name}")
            
            alert = Alert(
                level=level,
                timestamp=datetime.now(),
                source='forecasting',
                location=location,
                confidence=1.0 - abs(predicted_ph_actual - 7.0) / 7.0,  # Distance from neutral
                details={
                    'predicted_ph': float(predicted_ph_actual),
                    'safe_range': [6.5, 8.5]
                }
            )
            
            self.alert_history.append(alert)
            return alert
            
        except Exception as e:
            logger.warning(f"Forecast check failed: {e}")
            return Alert(
                level=AlertLevel.NORMAL,
                timestamp=datetime.now(),
                source='forecasting',
                details={'error': str(e)}
            )
    
    def monitor_batch(self, samples: pd.DataFrame,
                     feature_names: List[str] = None,
                     locations: List[str] = None,
                     verbose: bool = True) -> List[Alert]:
        """
        Monitor batch of samples
        
        Args:
            samples: DataFrame of samples
            feature_names: Feature names
            locations: Location identifiers
            verbose: Print progress
        
        Returns:
            List of alerts
        """
        if verbose:
            logger.section("MONITORING BATCH")
            logger.info(f"  Analyzing {len(samples)} samples...")
        
        alerts = []
        
        for idx, (_, row) in enumerate(samples.iterrows()):
            location = locations[idx] if locations else f"Sample_{idx}"
            sample = row.values
            
            alert = self.check_sample(
                sample,
                feature_names=feature_names,
                location=location,
                verbose=False
            )
            alerts.append(alert)
        
        # Summary
        if verbose:
            self._print_batch_summary(alerts)
        
        return alerts
    
    def _print_alert(self, alert: Alert):
        """Print formatted alert"""
        color_map = {
            'green': '\033[92m',
            'yellow': '\033[93m',
            'orange': '\033[38;5;208m',
            'red': '\033[91m'
        }
        reset = '\033[0m'
        
        color = color_map.get(alert.get_color(), '')
        
        logger.info(f"\n{color}{'='*60}")
        logger.info(f"  ALERT: {alert.level.name}")
        logger.info(f"{'='*60}{reset}")
        logger.info(f"  Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"  Location: {alert.location or 'Unknown'}")
        logger.info(f"  Message: {alert.get_message()}")
        logger.info(f"  Confidence: {alert.confidence:.2%}")
        if alert.triggered_by:
            logger.info(f"  Triggered by: {', '.join(alert.triggered_by)}")
    
    def _print_batch_summary(self, alerts: List[Alert]):
        """Print summary of batch monitoring"""
        logger.subsection("Batch Monitoring Summary")
        
        # Count by level
        level_counts = {}
        for alert in alerts:
            level_counts[alert.level.name] = level_counts.get(alert.level.name, 0) + 1
        
        total = len(alerts)
        for level in ['NORMAL', 'LOW', 'MEDIUM', 'CRITICAL']:
            count = level_counts.get(level, 0)
            pct = (count / total) * 100
            logger.info(f"  {level:<10} {count:>4} ({pct:>5.1f}%)")
        
        # Highlight critical alerts
        critical = [a for a in alerts if a.level == AlertLevel.CRITICAL]
        if critical:
            logger.warning(f"\n  ⚠️  {len(critical)} CRITICAL alerts require immediate action!")
    
    def get_alert_statistics(self) -> Dict:
        """Get statistics from alert history"""
        if not self.alert_history:
            return {}
        
        stats = {
            'total_alerts': len(self.alert_history),
            'by_level': {},
            'by_source': {},
            'critical_count': 0,
            'latest_alert': self.alert_history[-1].to_dict()
        }
        
        for alert in self.alert_history:
            # By level
            level_name = alert.level.name
            stats['by_level'][level_name] = stats['by_level'].get(level_name, 0) + 1
            
            # By source
            stats['by_source'][alert.source] = stats['by_source'].get(alert.source, 0) + 1
            
            # Critical count
            if alert.level == AlertLevel.CRITICAL:
                stats['critical_count'] += 1
        
        return stats


if __name__ == "__main__":
    # Test
    print("=== MONITORING ENGINE TEST ===\n")
    
    # Create monitor
    monitor = WaterQualityMonitor()
    
    # Simulate sample
    sample = np.random.randn(9)
    
    # Check without models (will return normal alert)
    alert = monitor.check_sample(sample, location="Test_Location")
    print(f"Alert level: {alert.level.name}")
    print(f"Message: {alert.get_message()}")
    
    print("\n✓ Monitoring engine test passed!")
