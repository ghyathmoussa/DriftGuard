"""
Logging Utilities Module

This module provides logging utilities for tracking performance metrics and
drift detection results in ML monitoring systems.
"""

from typing import Dict, Any
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    filename='logs/metrics_drift.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)


def log_metrics(metrics: Dict[str, Any], environment: str) -> None:
    """
    Log performance metrics to file.
    
    Records performance metrics with environment context to the configured log file.
    
    Args:
        metrics: Dictionary of performance metrics (e.g., accuracy, precision).
        environment: Environment identifier (e.g., "training" or "production").
    
    Example:
        >>> metrics = {"accuracy": 0.95, "f1_score": 0.92}
        >>> log_metrics(metrics, "production")
    """
    log_message = f"Environment: {environment}, Metrics: {metrics}"
    logging.info(log_message)


def log_drift(drift_results: Dict[str, Any]) -> None:
    """
    Log drift detection results to file.
    
    Records drift detection results including drift indicators and
    statistical metrics for each feature.
    
    Args:
        drift_results: Dictionary of drift detection results per feature.
                      Each feature maps to drift status and metrics.
    
    Example:
        >>> drift_results = {
        ...     "age": {"drift_detected": True, "p_value": 0.01}
        ... }
        >>> log_drift(drift_results)
    """
    log_message = f"Drift Results: {drift_results}"
    logging.info(log_message)