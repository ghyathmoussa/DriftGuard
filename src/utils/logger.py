# src/utils/logger.py

import logging
from datetime import datetime

# Configure logging
logging.basicConfig(filename='logs/metrics_drift.log', level=logging.INFO, format='%(asctime)s - %(message)s')

def log_metrics(metrics, environment):
    """
    Log performance metrics.
    
    Args:
        metrics (dict): Dictionary of performance metrics.
        environment (str): Environment (training or production).
    """
    log_message = f"Environment: {environment}, Metrics: {metrics}"
    logging.info(log_message)

def log_drift(drift_results):
    """
    Log drift detection results.
    
    Args:
        drift_results (dict): Dictionary of drift detection results.
    """
    log_message = f"Drift Results: {drift_results}"
    logging.info(log_message)