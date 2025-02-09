# alerting/alerts.py

import smtplib
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

class AlertSystem:
    def __init__(self, email_config=None, slack_config=None):
        """
        Initialize the alert system.
        
        Args:
            email_config (dict): Configuration for email alerts.
            slack_config (dict): Configuration for Slack alerts.
        """
        self.email_config = email_config
        self.slack_config = slack_config

    def send_email(self, subject, message):
        """
        Send an email alert.
        
        Args:
            subject (str): Email subject.
            message (str): Email body.
        """
        if self.email_config is None:
            raise ValueError("Email configuration not provided.")
        
        server = smtplib.SMTP(self.email_config["smtp_server"], self.email_config["smtp_port"])
        server.starttls()
        server.login(self.email_config["username"], self.email_config["password"])
        server.sendmail(self.email_config["from_email"], self.email_config["to_email"], f"Subject: {subject}\n\n{message}")
        server.quit()

    def send_slack_message(self, message):
        """
        Send a Slack alert.
        
        Args:
            message (str): Message to send.
        """
        if self.slack_config is None:
            raise ValueError("Slack configuration not provided.")
        
        client = WebClient(token=self.slack_config["token"])
        try:
            client.chat_postMessage(channel=self.slack_config["channel"], text=message)
        except SlackApiError as e:
            print(f"Slack API error: {e.response['error']}")