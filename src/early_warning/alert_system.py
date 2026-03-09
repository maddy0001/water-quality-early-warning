"""
============================================================================
Alert System - Alert Level Definitions
Defines 4-level alert system for water quality monitoring
============================================================================
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime


class AlertLevel(Enum):
    """Alert severity levels"""
    NORMAL = 0      # Safe, no action needed
    LOW = 1         # Minor deviation, monitor
    MEDIUM = 2      # Significant concern, investigate
    CRITICAL = 3    # Immediate action required


@dataclass
class Alert:
    """
    Single alert instance
    """
    level: AlertLevel
    timestamp: datetime
    source: str  # 'classification', 'forecasting', 'anomaly'
    location: Optional[str] = None
    confidence: float = 0.0
    triggered_by: List[str] = None
    details: Dict = None
    
    def __post_init__(self):
        if self.triggered_by is None:
            self.triggered_by = []
        if self.details is None:
            self.details = {}
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'level': self.level.name,
            'level_value': self.level.value,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source,
            'location': self.location,
            'confidence': float(self.confidence),
            'triggered_by': self.triggered_by,
            'details': self.details
        }
    
    def get_color(self) -> str:
        """Get color code for alert level"""
        colors = {
            AlertLevel.NORMAL: 'green',
            AlertLevel.LOW: 'yellow',
            AlertLevel.MEDIUM: 'orange',
            AlertLevel.CRITICAL: 'red'
        }
        return colors[self.level]
    
    def get_message(self) -> str:
        """Get human-readable alert message"""
        messages = {
            AlertLevel.NORMAL: "Water quality is normal",
            AlertLevel.LOW: "Minor water quality deviation detected",
            AlertLevel.MEDIUM: "Significant water quality concern - investigation recommended",
            AlertLevel.CRITICAL: "CRITICAL: Immediate action required - water may be contaminated"
        }
        return messages[self.level]


class AlertThresholds:
    """
    Thresholds for triggering different alert levels
    """
    
    # Classification confidence thresholds
    CLASSIFICATION_CRITICAL = 0.9   # 90%+ confidence of contamination
    CLASSIFICATION_MEDIUM = 0.7     # 70%+ confidence
    CLASSIFICATION_LOW = 0.5        # 50%+ confidence
    
    # Forecasting deviation thresholds (from normal pH range)
    FORECASTING_CRITICAL = 0.15     # ±0.15 pH units
    FORECASTING_MEDIUM = 0.10       # ±0.10 pH units
    FORECASTING_LOW = 0.05          # ±0.05 pH units
    
    # Anomaly detection vote thresholds
    ANOMALY_CRITICAL = 4            # All 4 detectors flag
    ANOMALY_MEDIUM = 3              # 3 detectors flag
    ANOMALY_LOW = 2                 # 2 detectors flag
    
    # pH safe range (WHO guidelines: 6.5-8.5)
    PH_MIN_SAFE = 6.5
    PH_MAX_SAFE = 8.5
    
    @classmethod
    def get_alert_level_from_classification(cls, confidence: float,
                                           predicted_class: int) -> AlertLevel:
        """
        Determine alert level from classification prediction
        
        Args:
            confidence: Model confidence (0-1)
            predicted_class: 0=non-potable, 1=potable
        
        Returns:
            AlertLevel
        """
        if predicted_class == 1:  # Potable
            return AlertLevel.NORMAL
        
        # Non-potable - check confidence
        if confidence >= cls.CLASSIFICATION_CRITICAL:
            return AlertLevel.CRITICAL
        elif confidence >= cls.CLASSIFICATION_MEDIUM:
            return AlertLevel.MEDIUM
        elif confidence >= cls.CLASSIFICATION_LOW:
            return AlertLevel.LOW
        else:
            return AlertLevel.NORMAL
    
    @classmethod
    def get_alert_level_from_forecasting(cls, predicted_ph: float) -> AlertLevel:
        """
        Determine alert level from pH forecast
        
        Args:
            predicted_ph: Forecasted pH value
        
        Returns:
            AlertLevel
        """
        # Check how far outside safe range
        if predicted_ph < cls.PH_MIN_SAFE:
            deviation = cls.PH_MIN_SAFE - predicted_ph
        elif predicted_ph > cls.PH_MAX_SAFE:
            deviation = predicted_ph - cls.PH_MAX_SAFE
        else:
            return AlertLevel.NORMAL
        
        # Classify based on deviation magnitude
        if deviation >= cls.FORECASTING_CRITICAL:
            return AlertLevel.CRITICAL
        elif deviation >= cls.FORECASTING_MEDIUM:
            return AlertLevel.MEDIUM
        elif deviation >= cls.FORECASTING_LOW:
            return AlertLevel.LOW
        else:
            return AlertLevel.NORMAL
    
    @classmethod
    def get_alert_level_from_anomaly(cls, vote_count: int) -> AlertLevel:
        """
        Determine alert level from anomaly detector votes
        
        Args:
            vote_count: Number of detectors flagging anomaly (0-4)
        
        Returns:
            AlertLevel
        """
        if vote_count >= cls.ANOMALY_CRITICAL:
            return AlertLevel.CRITICAL
        elif vote_count >= cls.ANOMALY_MEDIUM:
            return AlertLevel.MEDIUM
        elif vote_count >= cls.ANOMALY_LOW:
            return AlertLevel.LOW
        else:
            return AlertLevel.NORMAL


class AlertAggregator:
    """
    Aggregates alerts from multiple sources into final alert level
    Uses weighted voting
    """
    
    # Source weights (importance)
    WEIGHTS = {
        'classification': 0.4,
        'forecasting': 0.3,
        'anomaly': 0.3
    }
    
    @classmethod
    def aggregate(cls, alerts: List[Alert]) -> Alert:
        """
        Aggregate multiple alerts into single alert
        Uses weighted average of alert levels
        
        Args:
            alerts: List of alerts from different sources
        
        Returns:
            Aggregated alert
        """
        if not alerts:
            return Alert(
                level=AlertLevel.NORMAL,
                timestamp=datetime.now(),
                source='aggregated'
            )
        
        # Weighted sum of alert levels
        weighted_sum = 0.0
        total_weight = 0.0
        triggered_by = []
        all_details = {}
        
        for alert in alerts:
            weight = cls.WEIGHTS.get(alert.source, 0.2)
            weighted_sum += alert.level.value * weight
            total_weight += weight
            triggered_by.append(f"{alert.source}:{alert.level.name}")
            all_details[alert.source] = alert.details
        
        # Compute average alert level
        avg_level = weighted_sum / total_weight if total_weight > 0 else 0
        
        # Round to nearest level
        final_level_value = round(avg_level)
        final_level = AlertLevel(final_level_value)
        
        # If any source is CRITICAL, elevate to at least MEDIUM
        if any(a.level == AlertLevel.CRITICAL for a in alerts):
            if final_level.value < AlertLevel.MEDIUM.value:
                final_level = AlertLevel.MEDIUM
        
        return Alert(
            level=final_level,
            timestamp=datetime.now(),
            source='aggregated',
            confidence=avg_level / 3.0,  # Normalize to 0-1
            triggered_by=triggered_by,
            details=all_details
        )


if __name__ == "__main__":
    # Test
    print("=== ALERT SYSTEM TEST ===\n")
    
    # Test classification thresholds
    alert1 = Alert(
        level=AlertThresholds.get_alert_level_from_classification(0.95, 0),
        timestamp=datetime.now(),
        source='classification',
        confidence=0.95
    )
    print(f"Classification (0.95 conf, non-potable): {alert1.level.name}")
    
    # Test forecasting thresholds
    alert2 = Alert(
        level=AlertThresholds.get_alert_level_from_forecasting(5.8),
        timestamp=datetime.now(),
        source='forecasting',
        details={'predicted_ph': 5.8}
    )
    print(f"Forecasting (pH=5.8): {alert2.level.name}")
    
    # Test anomaly thresholds
    alert3 = Alert(
        level=AlertThresholds.get_alert_level_from_anomaly(3),
        timestamp=datetime.now(),
        source='anomaly',
        details={'vote_count': 3}
    )
    print(f"Anomaly (3 votes): {alert3.level.name}")
    
    # Test aggregation
    aggregated = AlertAggregator.aggregate([alert1, alert2, alert3])
    print(f"\nAggregated: {aggregated.level.name}")
    print(f"Triggered by: {aggregated.triggered_by}")
    print(f"Message: {aggregated.get_message()}")
    
    print("\n✓ Alert system test passed!")
