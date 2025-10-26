"""
Concept Drift Detection Module

This module provides a unified interface for detecting concept drift in streaming data
using various algorithms from the River library.
"""

from typing import Union, Literal
import numpy as np
from river.drift import ADWIN, KSWIN, PageHinkley

DriftMethod = Literal["ADWIN", "KSWIN", "PageHinkley"]


class ConceptDriftDetector:
    """
    A detector for concept drift in streaming data using various algorithms.
    
    Concept drift occurs when the statistical properties of the target variable
    change over time, which can degrade model performance. This detector monitors
    prediction errors and alerts when significant drift is detected.
    
    Attributes:
        detector: The underlying drift detection algorithm.
        method: The name of the drift detection method being used.
    
    Example:
        >>> detector = ConceptDriftDetector(method="ADWIN")
        >>> for true_val, pred_val in zip(y_true, y_pred):
        ...     if detector.update(true_val, pred_val):
        ...         print("Drift detected!")
        ...         detector.reset()
    """
    
    def __init__(self, method: DriftMethod = "ADWIN") -> None:
        """
        Initialize the concept drift detector.
        
        Args:
            method: Drift detection method. Options: "ADWIN", "KSWIN", "PageHinkley".
                   Default is "ADWIN".
        
        Raises:
            ValueError: If an invalid method is specified.
        """
        if method == "ADWIN":
            self.detector = ADWIN()
        elif method == "KSWIN":
            self.detector = KSWIN()
        elif method == "PageHinkley":
            self.detector = PageHinkley()
        else:
            raise ValueError("Invalid method. Choose from 'ADWIN', 'KSWIN', 'PageHinkley'.")
        
        self.method = method

    def update(self, y_true: Union[int, float], y_pred: Union[int, float]) -> bool:
        """
        Update the drift detector with new data point.
        
        This method compares the true and predicted values, calculates the error,
        and updates the internal drift detector state. It returns whether drift
        has been detected at this point.
        
        Args:
            y_true: Actual/true label or value.
            y_pred: Predicted label or value.
        
        Returns:
            True if drift is detected, False otherwise.
        
        Example:
            >>> detector = ConceptDriftDetector()
            >>> drift_detected = detector.update(y_true=1, y_pred=0)
            >>> if drift_detected:
            ...     print("Drift detected!")
        """
        error = int(y_true != y_pred)  # Binary error (0 or 1)
        self.detector.update(error)
        
        return self.detector.drift_detected

    def reset(self) -> None:
        """
        Reset the drift detector to its initial state.
        
        This method creates a new instance of the drift detector, clearing all
        historical data and statistics. This is typically done after drift is
        detected to start monitoring from a clean state.
        
        Example:
            >>> detector = ConceptDriftDetector(method="ADWIN")
            >>> if detector.update(y_true=1, y_pred=0):
            ...     detector.reset()  # Start fresh after detecting drift
        """
        if self.method == "ADWIN":
            self.detector = ADWIN()
        elif self.method == "KSWIN":
            self.detector = KSWIN()
        elif self.method == "PageHinkley":
            self.detector = PageHinkley()