"""
Performance Monitoring Module

This module provides comprehensive performance monitoring for ML models,
tracking metrics over time and comparing training vs. production performance.
"""

from typing import Dict, List, Optional, Union, Any
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from src.utils.logger import log_metrics


class PerformanceMonitor:
    """
    Monitor and track ML model performance metrics over time.
    
    This class computes classification metrics and maintains separate histories
    for training and production environments, enabling comparison and drift detection.
    
    Attributes:
        metrics_history: Dictionary storing metrics for training and production.
        metrics_names: List of metric names being tracked.
    
    Example:
        >>> monitor = PerformanceMonitor()
        >>> y_true = [0, 1, 1, 0]
        >>> y_pred = [0, 1, 0, 0]
        >>> monitor.update(y_true, y_pred, environment="production")
        >>> history = monitor.get_metrics_history()
    """
    
    def __init__(self) -> None:
        """
        Initialize the performance monitor.
        
        Sets up empty metrics histories for both training and production environments.
        """
        self.metrics_history = {
            "training": [],
            "production": []
        }
        self.metrics_names = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]

    def compute_metrics(
        self,
        y_true: Union[np.ndarray, List],
        y_pred: Union[np.ndarray, List],
        y_prob: Optional[Union[np.ndarray, List]] = None
    ) -> Dict[str, Optional[float]]:
        """
        Compute comprehensive classification performance metrics.
        
        Calculates standard classification metrics including accuracy, precision,
        recall, F1 score, and optionally ROC-AUC if probabilities are provided.
        
        Args:
            y_true: True labels as array-like.
            y_pred: Predicted labels as array-like.
            y_prob: Predicted probabilities for ROC-AUC calculation (optional).
        
        Returns:
            Dictionary containing computed metrics:
                - accuracy: Overall accuracy
                - precision: Weighted precision
                - recall: Weighted recall
                - f1_score: Weighted F1 score
                - roc_auc: ROC-AUC score (None if y_prob not provided)
        
        Example:
            >>> monitor = PerformanceMonitor()
            >>> metrics = monitor.compute_metrics([0, 1, 1], [0, 1, 0])
            >>> print(f"Accuracy: {metrics['accuracy']:.3f}")
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1_score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "roc_auc": roc_auc_score(y_true, y_prob) if y_prob is not None else None
        }
        return metrics

    def update(
        self,
        y_true: Union[np.ndarray, List],
        y_pred: Union[np.ndarray, List],
        y_prob: Optional[Union[np.ndarray, List]] = None,
        environment: str = "production"
    ) -> None:
        """
        Update the performance monitor with new data.
        
        Computes metrics for the provided predictions and stores them in the
        history with a timestamp for time-series tracking.
        
        Args:
            y_true: True labels as array-like.
            y_pred: Predicted labels as array-like.
            y_prob: Predicted probabilities for ROC-AUC (optional).
            environment: Environment to update ("training" or "production").
                        Default is "production".
        
        Example:
            >>> monitor = PerformanceMonitor()
            >>> monitor.update([0, 1], [0, 0], environment="production")
        """
        metrics = self.compute_metrics(y_true, y_pred, y_prob)
        metrics["timestamp"] = datetime.now()  # Add timestamp for tracking over time
        self.metrics_history[environment].append(metrics)
        log_metrics(metrics, environment)

    def get_metrics_history(self, environment: str = "production") -> pd.DataFrame:
        """
        Get the metrics history for a specific environment.
        
        Args:
            environment: Environment to retrieve metrics for ("training" or "production").
                        Default is "production".
        
        Returns:
            DataFrame containing the metrics history with timestamps.
        
        Example:
            >>> monitor = PerformanceMonitor()
            >>> monitor.update([0, 1], [0, 1])
            >>> history = monitor.get_metrics_history("production")
            >>> print(history.columns)
        """
        return pd.DataFrame(self.metrics_history[environment])

    def compare_performance(self) -> pd.DataFrame:
        """
        Compare training vs. production performance.
        
        Computes mean metrics for both environments and returns a comparison DataFrame.
        
        Returns:
            DataFrame with two columns ('training' and 'production') showing
            mean values for each metric.
        
        Raises:
            ValueError: If no metrics are available for comparison in either environment.
        
        Example:
            >>> monitor = PerformanceMonitor()
            >>> monitor.update([0, 1], [0, 1], environment="training")
            >>> monitor.update([0, 1], [0, 0], environment="production")
            >>> comparison = monitor.compare_performance()
            >>> print(comparison)
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

    def plot_metrics_over_time(self, environment: str = "production") -> None:
        """
        Plot metrics over time for a specific environment.
        
        Creates a time-series line plot showing the evolution of all tracked
        metrics in the specified environment.
        
        Args:
            environment: Environment to plot metrics for ("training" or "production").
                        Default is "production".
        
        Raises:
            ValueError: If no metrics are available for the specified environment.
        
        Example:
            >>> monitor = PerformanceMonitor()
            >>> for _ in range(5):
            ...     monitor.update([0, 1], [0, 1])
            >>> monitor.plot_metrics_over_time("production")
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
