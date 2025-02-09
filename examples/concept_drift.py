import numpy as np
from src.concept_drift.drift_detection import ConceptDriftDetector
import matplotlib.pyplot as plt

# Simulate model predictions and actual labels
np.random.seed(42)
y_true = np.concatenate([np.zeros(500), np.ones(500)])  # No drift for first 500, then drift
y_pred = np.concatenate([np.zeros(500), np.zeros(500)])  # Model predicts 0 always

# Initialize drift detector
drift_detector = ConceptDriftDetector(method="ADWIN")

# Monitor for drift
drift_points = []
for i, (true, pred) in enumerate(zip(y_true, y_pred)):
    drift_detected = drift_detector.update(true, pred)
    if drift_detected:
        print(f"Drift detected at index {i}!")
        drift_points.append(i)


plt.figure(figsize=(10, 6))
plt.plot(y_true, label="True Labels", color="blue", alpha=0.6)
plt.plot(y_pred, label="Predicted Labels", color="red", alpha=0.6)
for point in drift_points:
    plt.axvline(x=point, color="black", linestyle="--", label="Drift Detected" if point == drift_points[0] else "")
plt.title("Concept Drift Detection")
plt.xlabel("Time")
plt.ylabel("Label")
plt.legend()
plt.show()