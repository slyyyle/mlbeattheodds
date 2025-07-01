import os
import sys
import json
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import requests
from dataclasses import dataclass

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class NotificationChannel:
    """Configuration for a notification channel."""
    name: str
    enabled: bool
    config: Dict[str, Any]


@dataclass
class Alert:
    """Alert data structure."""
    id: str
    type: str
    severity: str
    message: str
    timestamp: datetime
    data: Dict[str, Any]
    channels: List[str]


class NotificationSystem:
    """
    Comprehensive notification system for MLB betting analytics.
    
    Features:
    - Multiple notification channels (email, Slack, webhook, SMS)
    - Alert prioritization and filtering
    - Rate limiting to prevent spam
    - Template-based messaging
    - Delivery tracking and retry logic
    """
    
    def __init__(self, config_file: str = "config/notifications.json"):
        self.config_file = Path(config_file)
        self.channels = {}
        self.alert_history = []
        self.delivery_log = []
        
        # Rate limiting
        self.rate_limits = {
            'high': {'max_per_hour': 10, 'sent_count': 0, 'reset_time': datetime.now()},
            'medium': {'max_per_hour': 5, 'sent_count': 0, 'reset_time': datetime.now()},
            'low': {'max_per_hour': 2, 'sent_count': 0, 'reset_time': datetime.now()}
        }
        
        # Load configuration
        self._load_configuration()
        
        # Initialize channels
        self._initialize_channels()
    
    def _load_configuration(self):
        """Load notification configuration from file."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    
                for channel_config in config.get('channels', []):
                    channel = NotificationChannel(
                        name=channel_config['name'],
                        enabled=channel_config.get('enabled', True),
                        config=channel_config.get('config', {})
                    )
                    self.channels[channel.name] = channel
                    
            except Exception as e:
                logger.error(f"Failed to load notification config: {e}")
                self._create_default_config()
        else:
            self._create_default_config()
    
    def _create_default_config(self):
        """Create default notification configuration."""
        default_config = {
            "channels": [
                {
                    "name": "email",
                    "enabled": False,
                    "config": {
                        "smtp_server": "smtp.gmail.com",
                        "smtp_port": 587,
                        "username": "",
                        "password": "",
                        "from_email": "",
                        "to_emails": []
                    }
                },
                {
                    "name": "slack",
                    "enabled": False,
                    "config": {
                        "webhook_url": "",
                        "channel": "#alerts",
                        "username": "MLB Betting Bot"
                    }
                },
                {
                    "name": "webhook",
                    "enabled": False,
                    "config": {
                        "url": "",
                        "method": "POST",
                        "headers": {},
                        "timeout": 30
                    }
                },
                {
                    "name": "console",
                    "enabled": True,
                    "config": {}
                }
            ]
        }
        
        # Create config directory
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save default config
        with open(self.config_file, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        logger.info(f"Created default notification config at {self.config_file}")
    
    def _initialize_channels(self):
        """Initialize notification channels."""
        for channel_name, channel in self.channels.items():
            if channel.enabled:
                logger.info(f"Initialized notification channel: {channel_name}")
    
    def send_alert(self, alert_type: str, severity: str, message: str, 
                  data: Dict[str, Any] = None, channels: List[str] = None) -> str:
        """
        Send an alert through configured notification channels.
        
        Args:
            alert_type: Type of alert (e.g., 'betting_opportunity', 'system_error')
            severity: Alert severity ('high', 'medium', 'low')
            message: Alert message
            data: Additional alert data
            channels: Specific channels to use (if None, uses all enabled channels)
            
        Returns:
            Alert ID
        """
        # Generate alert ID
        alert_id = f"{alert_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.alert_history)}"
        
        # Create alert object
        alert = Alert(
            id=alert_id,
            type=alert_type,
            severity=severity,
            message=message,
            timestamp=datetime.now(),
            data=data or {},
            channels=channels or list(self.channels.keys())
        )
        
        # Check rate limits
        if not self._check_rate_limit(severity):
            logger.warning(f"Rate limit exceeded for {severity} alerts, skipping alert: {alert_id}")
            return alert_id
        
        # Store alert
        self.alert_history.append(alert)
        
        # Send through channels
        self._send_through_channels(alert)
        
        # Update rate limit counters
        self._update_rate_limit(severity)
        
        return alert_id
    
    def _check_rate_limit(self, severity: str) -> bool:
        """Check if alert can be sent based on rate limits."""
        now = datetime.now()
        rate_limit = self.rate_limits.get(severity, self.rate_limits['low'])
        
        # Reset counter if hour has passed
        if now - rate_limit['reset_time'] > timedelta(hours=1):
            rate_limit['sent_count'] = 0
            rate_limit['reset_time'] = now
        
        return rate_limit['sent_count'] < rate_limit['max_per_hour']
    
    def _update_rate_limit(self, severity: str):
        """Update rate limit counter."""
        if severity in self.rate_limits:
            self.rate_limits[severity]['sent_count'] += 1
    
    def _send_through_channels(self, alert: Alert):
        """Send alert through all specified channels."""
        for channel_name in alert.channels:
            if channel_name in self.channels and self.channels[channel_name].enabled:
                try:
                    success = self._send_to_channel(alert, channel_name)
                    
                    # Log delivery attempt
                    self.delivery_log.append({
                        'alert_id': alert.id,
                        'channel': channel_name,
                        'timestamp': datetime.now(),
                        'success': success
                    })
                    
                except Exception as e:
                    logger.error(f"Failed to send alert {alert.id} to {channel_name}: {e}")
                    
                    self.delivery_log.append({
                        'alert_id': alert.id,
                        'channel': channel_name,
                        'timestamp': datetime.now(),
                        'success': False,
                        'error': str(e)
                    })
    
    def _send_to_channel(self, alert: Alert, channel_name: str) -> bool:
        """Send alert to specific channel."""
        channel = self.channels[channel_name]
        
        if channel_name == 'email':
            return self._send_email(alert, channel)
        elif channel_name == 'slack':
            return self._send_slack(alert, channel)
        elif channel_name == 'webhook':
            return self._send_webhook(alert, channel)
        elif channel_name == 'console':
            return self._send_console(alert, channel)
        else:
            logger.warning(f"Unknown channel type: {channel_name}")
            return False
    
    def _send_email(self, alert: Alert, channel: NotificationChannel) -> bool:
        """Send alert via email."""
        try:
            config = channel.config
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = config['from_email']
            msg['To'] = ', '.join(config['to_emails'])
            msg['Subject'] = f"MLB Betting Alert - {alert.severity.upper()}: {alert.type}"
            
            # Email body
            body = self._format_email_body(alert)
            msg.attach(MIMEText(body, 'html'))
            
            # Send email
            server = smtplib.SMTP(config['smtp_server'], config['smtp_port'])
            server.starttls()
            server.login(config['username'], config['password'])
            
            text = msg.as_string()
            server.sendmail(config['from_email'], config['to_emails'], text)
            server.quit()
            
            logger.info(f"Email sent for alert {alert.id}")
            return True
            
        except Exception as e:
            logger.error(f"Email sending failed for alert {alert.id}: {e}")
            return False
    
    def _send_slack(self, alert: Alert, channel: NotificationChannel) -> bool:
        """Send alert via Slack webhook."""
        try:
            config = channel.config
            
            # Format Slack message
            slack_message = self._format_slack_message(alert)
            
            # Send to Slack
            response = requests.post(
                config['webhook_url'],
                json=slack_message,
                timeout=30
            )
            
            response.raise_for_status()
            logger.info(f"Slack message sent for alert {alert.id}")
            return True
            
        except Exception as e:
            logger.error(f"Slack sending failed for alert {alert.id}: {e}")
            return False
    
    def _send_webhook(self, alert: Alert, channel: NotificationChannel) -> bool:
        """Send alert via generic webhook."""
        try:
            config = channel.config
            
            # Prepare payload
            payload = {
                'alert_id': alert.id,
                'type': alert.type,
                'severity': alert.severity,
                'message': alert.message,
                'timestamp': alert.timestamp.isoformat(),
                'data': alert.data
            }
            
            # Send webhook
            response = requests.request(
                method=config.get('method', 'POST'),
                url=config['url'],
                json=payload,
                headers=config.get('headers', {}),
                timeout=config.get('timeout', 30)
            )
            
            response.raise_for_status()
            logger.info(f"Webhook sent for alert {alert.id}")
            return True
            
        except Exception as e:
            logger.error(f"Webhook sending failed for alert {alert.id}: {e}")
            return False
    
    def _send_console(self, alert: Alert, channel: NotificationChannel) -> bool:
        """Send alert to console/log."""
        try:
            log_message = f"ALERT [{alert.severity.upper()}] {alert.type}: {alert.message}"
            
            if alert.severity == 'high':
                logger.error(log_message)
            elif alert.severity == 'medium':
                logger.warning(log_message)
            else:
                logger.info(log_message)
            
            return True
            
        except Exception as e:
            logger.error(f"Console logging failed for alert {alert.id}: {e}")
            return False
    
    def _format_email_body(self, alert: Alert) -> str:
        """Format email body for alert."""
        severity_colors = {
            'high': '#FF4444',
            'medium': '#FF8800',
            'low': '#44AA44'
        }
        
        color = severity_colors.get(alert.severity, '#666666')
        
        html_body = f"""
        <html>
        <body>
            <h2 style="color: {color};">MLB Betting Alert - {alert.severity.upper()}</h2>
            
            <table style="border-collapse: collapse; width: 100%;">
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;"><strong>Alert Type</strong></td>
                    <td style="border: 1px solid #ddd; padding: 8px;">{alert.type}</td>
                </tr>
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;"><strong>Severity</strong></td>
                    <td style="border: 1px solid #ddd; padding: 8px; color: {color};">{alert.severity.upper()}</td>
                </tr>
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;"><strong>Timestamp</strong></td>
                    <td style="border: 1px solid #ddd; padding: 8px;">{alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</td>
                </tr>
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;"><strong>Message</strong></td>
                    <td style="border: 1px solid #ddd; padding: 8px;">{alert.message}</td>
                </tr>
            </table>
        """
        
        # Add data section if available
        if alert.data:
            html_body += """
            <h3>Additional Data</h3>
            <table style="border-collapse: collapse; width: 100%;">
            """
            
            for key, value in alert.data.items():
                html_body += f"""
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;"><strong>{key}</strong></td>
                    <td style="border: 1px solid #ddd; padding: 8px;">{value}</td>
                </tr>
                """
            
            html_body += "</table>"
        
        html_body += """
            <br>
            <p><em>This is an automated alert from the MLB Betting Analytics System.</em></p>
        </body>
        </html>
        """
        
        return html_body
    
    def _format_slack_message(self, alert: Alert) -> Dict[str, Any]:
        """Format Slack message for alert."""
        severity_colors = {
            'high': 'danger',
            'medium': 'warning',
            'low': 'good'
        }
        
        color = severity_colors.get(alert.severity, 'good')
        
        # Create attachment with alert details
        attachment = {
            'color': color,
            'title': f"MLB Betting Alert - {alert.type}",
            'text': alert.message,
            'fields': [
                {
                    'title': 'Severity',
                    'value': alert.severity.upper(),
                    'short': True
                },
                {
                    'title': 'Timestamp',
                    'value': alert.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    'short': True
                }
            ],
            'footer': 'MLB Betting Analytics',
            'ts': int(alert.timestamp.timestamp())
        }
        
        # Add data fields
        if alert.data:
            for key, value in list(alert.data.items())[:5]:  # Limit to 5 fields
                attachment['fields'].append({
                    'title': key.replace('_', ' ').title(),
                    'value': str(value),
                    'short': True
                })
        
        return {
            'username': self.channels['slack'].config.get('username', 'MLB Betting Bot'),
            'channel': self.channels['slack'].config.get('channel', '#alerts'),
            'attachments': [attachment]
        }
    
    def send_betting_opportunity(self, game_id: str, bet_type: str, 
                               expected_value: float, confidence: float,
                               recommended_stake: float, reasoning: str) -> str:
        """Send betting opportunity alert."""
        severity = 'high' if expected_value > 0.05 else 'medium' if expected_value > 0.02 else 'low'
        
        message = f"Betting opportunity found: {game_id} ({bet_type}) - {expected_value:.1%} EV"
        
        data = {
            'game_id': game_id,
            'bet_type': bet_type,
            'expected_value': f"{expected_value:.1%}",
            'confidence': f"{confidence:.1%}",
            'recommended_stake': f"{recommended_stake:.1%}",
            'reasoning': reasoning
        }
        
        return self.send_alert('betting_opportunity', severity, message, data)
    
    def send_system_error(self, component: str, error_message: str, 
                         error_details: Dict[str, Any] = None) -> str:
        """Send system error alert."""
        message = f"System error in {component}: {error_message}"
        
        data = {
            'component': component,
            'error_message': error_message
        }
        
        if error_details:
            data.update(error_details)
        
        return self.send_alert('system_error', 'high', message, data)
    
    def send_performance_alert(self, model_name: str, metric: str, 
                             current_value: float, threshold: float) -> str:
        """Send model performance degradation alert."""
        message = f"Performance degradation detected: {model_name} {metric} = {current_value:.4f} (threshold: {threshold:.4f})"
        
        data = {
            'model_name': model_name,
            'metric': metric,
            'current_value': current_value,
            'threshold': threshold,
            'degradation': f"{((threshold - current_value) / threshold * 100):.1f}%"
        }
        
        return self.send_alert('performance_degradation', 'medium', message, data)
    
    def send_daily_summary(self, summary_data: Dict[str, Any]) -> str:
        """Send daily performance summary."""
        message = f"Daily Summary - ROI: {summary_data.get('roi', 0):.2%}, Bets: {summary_data.get('total_bets', 0)}"
        
        return self.send_alert('daily_summary', 'low', message, summary_data)
    
    def get_alert_history(self, hours_back: int = 24) -> List[Alert]:
        """Get alert history for the last N hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        return [
            alert for alert in self.alert_history
            if alert.timestamp > cutoff_time
        ]
    
    def get_delivery_stats(self) -> Dict[str, Any]:
        """Get notification delivery statistics."""
        total_deliveries = len(self.delivery_log)
        successful_deliveries = len([d for d in self.delivery_log if d['success']])
        
        # Stats by channel
        channel_stats = {}
        for channel_name in self.channels.keys():
            channel_deliveries = [d for d in self.delivery_log if d['channel'] == channel_name]
            channel_successes = [d for d in channel_deliveries if d['success']]
            
            channel_stats[channel_name] = {
                'total': len(channel_deliveries),
                'successful': len(channel_successes),
                'success_rate': len(channel_successes) / len(channel_deliveries) if channel_deliveries else 0
            }
        
        return {
            'total_deliveries': total_deliveries,
            'successful_deliveries': successful_deliveries,
            'overall_success_rate': successful_deliveries / total_deliveries if total_deliveries else 0,
            'channel_stats': channel_stats
        }


def main():
    """Example usage of the notification system."""
    logger.info("Notification System - Example Usage")
    
    # Initialize notification system
    notifier = NotificationSystem()
    
    # Send various types of alerts
    
    # Betting opportunity
    notifier.send_betting_opportunity(
        game_id="NYY_vs_BOS",
        bet_type="moneyline",
        expected_value=0.045,
        confidence=0.75,
        recommended_stake=0.03,
        reasoning="Model identifies value due to line movement and recent form"
    )
    
    # System error
    notifier.send_system_error(
        component="data_ingestion",
        error_message="API rate limit exceeded",
        error_details={
            'api_calls_remaining': 0,
            'reset_time': '2024-01-15 14:00:00'
        }
    )
    
    # Performance alert
    notifier.send_performance_alert(
        model_name="ensemble_v1",
        metric="auc",
        current_value=0.52,
        threshold=0.55
    )
    
    # Daily summary
    notifier.send_daily_summary({
        'date': '2024-01-15',
        'total_bets': 5,
        'winning_bets': 3,
        'win_rate': 0.60,
        'roi': 0.08,
        'profit_loss': 400
    })
    
    # Get statistics
    stats = notifier.get_delivery_stats()
    logger.info(f"Delivery Stats: {stats}")
    
    # Get recent alerts
    recent_alerts = notifier.get_alert_history(24)
    logger.info(f"Recent alerts: {len(recent_alerts)}")


if __name__ == "__main__":
    main() 