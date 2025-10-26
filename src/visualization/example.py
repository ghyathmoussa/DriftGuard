"""
Dashboard Example

This script demonstrates how to set up and run the visualization dashboard
with simulated performance monitoring data and drift results.
"""

from typing import Dict, Any
import numpy as np
from src.monitoring.monitoring import PerformanceMonitor
from dashboard import VisualizationDashboard


def main() -> None:
    """
    Main function to set up and run the dashboard example.
    
    Creates a PerformanceMonitor, simulates some monitoring data,
    and launches the visualization dashboard.
    """
    # Simulate performance monitoring data
    performance_monitor = PerformanceMonitor()
    for _ in range(10):
        y_true = np.random.randint(0, 2, 100)
        y_pred = np.random.randint(0, 2, 100)
        y_prob = np.random.rand(100)
        performance_monitor.update(y_true, y_pred, y_prob, environment="production")

    # Simulate drift results
    drift_results: Dict[str, Dict[str, float]] = {
        "feature1": {"wasserstein_distance": 0.5},
        "feature2": {"js_divergence": 0.3},
        "feature3": {"wasserstein_distance": 0.7}
    }

    # Initialize and run the dashboard
    dashboard = VisualizationDashboard(performance_monitor, drift_results)
    dashboard.run()


if __name__ == "__main__":
    main()