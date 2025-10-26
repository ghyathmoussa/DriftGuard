"""
Alerting System Module

This module provides a unified alerting system supporting multiple notification
channels including email and Slack for ML monitoring alerts.
"""

from typing import Dict, Optional, Any
import smtplib
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError


class AlertSystem:
    """
    Multi-channel alert system for ML monitoring.
    
    This class provides a unified interface for sending alerts through various
    channels (email, Slack) when drift or performance issues are detected.
    
    Attributes:
        email_config: Configuration dictionary for email alerts.
        slack_config: Configuration dictionary for Slack alerts.
    
    Example:
        >>> email_cfg = {
        ...     "smtp_server": "smtp.gmail.com",
        ...     "smtp_port": 587,
        ...     "username": "user@example.com",
        ...     "password": "password",
        ...     "from_email": "alerts@example.com",
        ...     "to_email": "team@example.com"
        ... }
        >>> alerts = AlertSystem(email_config=email_cfg)
        >>> alerts.send_email("Drift Alert", "Drift detected in feature X")
    """
    
    def __init__(
        self,
        email_config: Optional[Dict[str, Any]] = None,
        slack_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize the alert system.
        
        Args:
            email_config: Configuration for email alerts containing keys:
                         smtp_server, smtp_port, username, password,
                         from_email, to_email.
            slack_config: Configuration for Slack alerts containing keys:
                         token, channel.
        """
        self.email_config = email_config
        self.slack_config = slack_config

    def send_email(self, subject: str, message: str) -> None:
        """
        Send an email alert via SMTP.
        
        Connects to the configured SMTP server and sends an email with the
        specified subject and message.
        
        Args:
            subject: Email subject line.
            message: Email body content.
        
        Raises:
            ValueError: If email configuration is not provided.
            smtplib.SMTPException: If there's an error sending the email.
        
        Example:
            >>> alerts = AlertSystem(email_config=config)
            >>> alerts.send_email("Alert", "Model accuracy dropped below threshold")
        """
        if self.email_config is None:
            raise ValueError("Email configuration not provided.")
        
        server = smtplib.SMTP(self.email_config["smtp_server"], self.email_config["smtp_port"])
        server.starttls()
        server.login(self.email_config["username"], self.email_config["password"])
        server.sendmail(self.email_config["from_email"], self.email_config["to_email"], f"Subject: {subject}\n\n{message}")
        server.quit()

    def send_slack_message(self, message: str) -> None:
        """
        Send a Slack alert to the configured channel.
        
        Uses the Slack SDK to post a message to the specified channel.
        
        Args:
            message: Message content to send to Slack.
        
        Raises:
            ValueError: If Slack configuration is not provided.
            SlackApiError: If there's an error communicating with Slack API.
        
        Example:
            >>> slack_cfg = {"token": "xoxb-...", "channel": "#alerts"}
            >>> alerts = AlertSystem(slack_config=slack_cfg)
            >>> alerts.send_slack_message("Drift detected in production model")
        """
        if self.slack_config is None:
            raise ValueError("Slack configuration not provided.")
        
        client = WebClient(token=self.slack_config["token"])
        try:
            client.chat_postMessage(channel=self.slack_config["channel"], text=message)
        except SlackApiError as e:
            print(f"Slack API error: {e.response['error']}")