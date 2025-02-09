import numpy as np
from river.drift import ADWIN, DDMS, PageHinkley

class ConceptDriftDetector:
    def __init__(self, method="ADWIN"):
        """
        Initialize the concept drift detector.
        
        Args:
            method (str): Drift detection method. Options: "ADWIN", "DDM", "PageHinkley".
        """
        if method == "ADWIN":
            self.detector = ADWIN()
        elif method == "DDM":
            self.detector = DDM()
        elif method == "PageHinkley":
            self.detector = PageHinkley()
        else:
            raise ValueError("Invalid method. Choose from 'ADWIN', 'DDM', 'PageHinkley'.")
        
        self.method = method

    def update(self, y_true, y_pred):
        """
        Update the drift detector with new data.
        
        Args:
            y_true (float or int): Actual label.
            y_pred (float or int): Predicted label.
        
        Returns:
            bool: True if drift is detected, False otherwise.
        """
        error = int(y_true != y_pred)  # Binary error (0 or 1)
        self.detector.update(error)
        
        return self.detector.drift_detected

    def reset(self):
        """
        Reset the drift detector.
        """
        if self.method == "ADWIN":
            self.detector = ADWIN()
        elif self.method == "DDM":
            self.detector = DDM()
        elif self.method == "PageHinkley":
            self.detector = PageHinkley()