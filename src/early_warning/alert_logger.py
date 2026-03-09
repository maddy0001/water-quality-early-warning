"""
============================================================================
Alert Logger and Notification System
Logs alerts and triggers notifications
============================================================================
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime, timedelta

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import LOGS_DIR, METRICS_DIR
from utils.logger import get_logger
from early_warning.alert_system import Alert, AlertLevel

logger = get_logger()


class AlertLogger:
    """
    Persistent storage for alerts
    Logs to JSON and provides query/reporting
    """

    def __init__(self, log_dir: Path = None):
        """Initialize alert logger"""
        self.log_dir = log_dir or (LOGS_DIR / 'alerts')
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.current_session_file = self.log_dir / f"alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.alerts_buffer = []

    def log_alert(self, alert: Alert):
        """Log single alert"""
        self.alerts_buffer.append(alert.to_dict())
    
    def log_batch(self, alerts: List[Alert]):
        """Log multiple alerts"""
        for alert in alerts:
            self.log_alert(alert)
    
    def flush(self):
        """Write buffered alerts to disk"""
        if not self.alerts_buffer:
            return
        
        # Append to session file
        if self.current_session_file.exists():
            with open(self.current_session_file, 'r') as f:
                existing = json.load(f)
        else:
            existing = {'alerts': []}
        
        existing['alerts'].extend(self.alerts_buffer)
        existing['last_updated'] = datetime.now().isoformat()
        existing['total_alerts'] = len(existing['alerts'])
        
        with open(self.current_session_file, 'w') as f:
            json.dump(existing, f, indent=2)
        
        logger.success(f"  Logged {len(self.alerts_buffer)} alerts to {self.current_session_file.name}")
        self.alerts_buffer = []
    
    def get_recent_alerts(self, hours: int = 24,
                         level: AlertLevel = None) -> List[Dict]:
        """
        Get recent alerts
        
        Args:
            hours: Look back this many hours
            level: Filter by alert level
        
        Returns:
            List of alert dictionaries
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        
        recent_alerts = []
        
        # Read all session files
        for log_file in sorted(self.log_dir.glob("alerts_*.json")):
            try:
                with open(log_file, 'r') as f:
                    data = json.load(f)
                
                for alert in data.get('alerts', []):
                    alert_time = datetime.fromisoformat(alert['timestamp'])
                    
                    # Check time filter
                    if alert_time < cutoff:
                        continue
                    
                    # Check level filter
                    if level and alert['level'] != level.name:
                        continue
                    
                    recent_alerts.append(alert)
            except:
                pass
        
        return recent_alerts
    
    def generate_report(self, hours: int = 24) -> Dict:
        """
        Generate alert statistics report
        
        Args:
            hours: Report period
        
        Returns:
            Statistics dictionary
        """
        alerts = self.get_recent_alerts(hours=hours)
        
        if not alerts:
            return {
                'period_hours': hours,
                'total_alerts': 0,
                'message': 'No alerts in this period'
            }
        
        # Count by level
        by_level = {}
        by_source = {}
        by_location = {}
        
        for alert in alerts:
            level = alert['level']
            source = alert['source']
            location = alert.get('location', 'Unknown')
            
            by_level[level] = by_level.get(level, 0) + 1
            by_source[source] = by_source.get(source, 0) + 1
            by_location[location] = by_location.get(location, 0) + 1
        
        # Find critical alerts
        critical_alerts = [a for a in alerts if a['level'] == 'CRITICAL']
        
        report = {
            'period_hours': hours,
            'total_alerts': len(alerts),
            'by_level': by_level,
            'by_source': by_source,
            'by_location': by_location,
            'critical_count': len(critical_alerts),
            'critical_alerts': critical_alerts[-5:] if critical_alerts else [],  # Last 5
            'latest_alert': alerts[-1] if alerts else None
        }
        
        return report


class NotificationSystem:
    """
    Notification system for critical alerts
    (Placeholder for email/SMS/webhook integration)
    """

    def __init__(self):
        """Initialize notification system"""
        self.enabled = False  # Disabled by default
        self.notification_log = []
    
    def send_notification(self, alert: Alert) -> bool:
        """
        Send notification for alert
        
        In production, this would:
        - Send email to operators
        - Trigger SMS alerts
        - Post to webhook/Slack
        - Update dashboard
        
        Args:
            alert: Alert to notify about
        
        Returns:
            Success status
        """
        if not self.enabled:
            return False
        
        # Simulate notification
        notification = {
            'timestamp': datetime.now().isoformat(),
            'alert_level': alert.level.name,
            'location': alert.location,
            'message': alert.get_message(),
            'details': alert.details
        }
        
        self.notification_log.append(notification)
        
        logger.info(f"  📧 Notification sent: {alert.level.name} at {alert.location}")
        
        return True
    
    def notify_if_critical(self, alert: Alert) -> bool:
        """
        Auto-notify for critical alerts
        
        Args:
            alert: Alert to check
        
        Returns:
            Whether notification was sent
        """
        if alert.level in (AlertLevel.CRITICAL, AlertLevel.MEDIUM):
            return self.send_notification(alert)
        return False


def generate_dashboard_data(alert_logger: AlertLogger,
                           hours: int = 24) -> Dict:
    """
    Generate data for dashboard visualization
    
    Args:
        alert_logger: AlertLogger instance
        hours: Report period
    
    Returns:
        Dashboard data dictionary
    """
    report = alert_logger.generate_report(hours=hours)
    
    # Calculate alert rate
    alert_rate = report['total_alerts'] / hours if hours > 0 else 0
    
    # Risk score (0-100)
    risk_score = 0
    if 'by_level' in report:
        risk_score = (
            report['by_level'].get('CRITICAL', 0) * 40 +
            report['by_level'].get('MEDIUM', 0) * 20 +
            report['by_level'].get('LOW', 0) * 5
        )
        risk_score = min(risk_score, 100)
    
    dashboard = {
        'timestamp': datetime.now().isoformat(),
        'period_hours': hours,
        'summary': {
            'total_alerts': report['total_alerts'],
            'critical_alerts': report['critical_count'],
            'alert_rate_per_hour': round(alert_rate, 2),
            'risk_score': risk_score
        },
        'breakdown': report['by_level'],
        'sources': report['by_source'],
        'locations': report.get('by_location', {}),
        'latest_critical': report.get('critical_alerts', [])
    }
    
    return dashboard


if __name__ == "__main__":
    from early_warning.alert_system import Alert, AlertLevel
    
    print("=== ALERT LOGGER TEST ===\n")
    
    # Create logger
    alert_logger = AlertLogger()
    
    # Create test alerts
    alerts = [
        Alert(AlertLevel.NORMAL, datetime.now(), 'test', location='Site_A'),
        Alert(AlertLevel.LOW, datetime.now(), 'test', location='Site_B'),
        Alert(AlertLevel.CRITICAL, datetime.now(), 'test', location='Site_C', confidence=0.95)
    ]
    
    # Log alerts
    alert_logger.log_batch(alerts)
    alert_logger.flush()
    
    # Generate report
    report = alert_logger.generate_report(hours=1)
    print(f"Total alerts: {report['total_alerts']}")
    print(f"By level: {report['by_level']}")
    
    # Test notification system
    notifier = NotificationSystem()
    notifier.enabled = True
    notifier.notify_if_critical(alerts[2])
    print(f"Notifications sent: {len(notifier.notification_log)}")
    
    # Generate dashboard data
    dashboard = generate_dashboard_data(alert_logger, hours=24)
    print(f"Risk score: {dashboard['summary']['risk_score']}")
    
    print("\n✓ Alert logger test passed!")
