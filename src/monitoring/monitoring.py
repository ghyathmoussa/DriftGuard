# performance_monitoring.py

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from src.utils.logger import log_metrics

class PerformanceMonitor:
    def __init__(self):
        """
        Initialize the performance monitor.
        """
        self.metrics_history = {
            "training": [],
            "production": []
        }
        self.metrics_names = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]

    def compute_metrics(self, y_true, y_pred, y_prob=None):
        """
        Compute performance metrics.
        
        Args:
            y_true (array-like): True labels.
            y_pred (array-like): Predicted labels.
            y_prob (array-like): Predicted probabilities (for ROC-AUC).
        
        Returns:
            dict: Dictionary of computed metrics.
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1_score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "roc_auc": roc_auc_score(y_true, y_prob) if y_prob is not None else None
        }
        return metrics

    def update(self, y_true, y_pred, y_prob=None, environment="production"):
        """
        Update the performance monitor with new data.
        
        Args:
            y_true (array-like): True labels.
            y_pred (array-like): Predicted labels.
            y_prob (array-like): Predicted probabilities (for ROC-AUC).
            environment (str): Environment to update ("training" or "production").
        """
        metrics = self.compute_metrics(y_true, y_pred, y_prob)
        metrics["timestamp"] = datetime.now()  # Add timestamp for tracking over time
        self.metrics_history[environment].append(metrics)
        log_metrics(metrics, environment)

    def get_metrics_history(self, environment="production"):
        """
        Get the metrics history for a specific environment.
        
        Args:
            environment (str): Environment to retrieve metrics for ("training" or "production").
        
        Returns:
            pd.DataFrame: DataFrame containing the metrics history.
        """
        return pd.DataFrame(self.metrics_history[environment])

    def compare_performance(self):
        """
        Compare training vs. production performance.
        
        Returns:
            pd.DataFrame: DataFrame containing the comparison.
        """
        training_metrics = pd.DataFrame(self.metrics_history["training"])
        production_metrics = pd.DataFrame(self.metrics_history["production"])
        
        if len(training_metrics) == 0 or len(production_metrics) == 0:
            raise ValueError("No metrics available for comparison.")
        
        # Compute mean metrics for training and production
        comparison = pd.DataFrame({
            "training": training_metrics.mean(),
            "production": production_metrics.mean()
        })
        return comparison

    def plot_metrics_over_time(self, environment="production"):
        """
        Plot metrics over time for a specific environment.
        
        Args:
            environment (str): Environment to plot metrics for ("training" or "production").
        """
        metrics_df = self.get_metrics_history(environment)
        if len(metrics_df) == 0:
            raise ValueError(f"No metrics available for {environment} environment.")
        
        plt.figure(figsize=(12, 8))
        for metric in self.metrics_names:
            if metric in metrics_df.columns:
                plt.plot(metrics_df["timestamp"], metrics_df[metric], label=metric)
        plt.title(f"Performance Metrics Over Time ({environment.capitalize()})")
        plt.xlabel("Time")
        plt.ylabel("Metric Value")
        plt.legend()
        plt.grid()
        plt.show()
