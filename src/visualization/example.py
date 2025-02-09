 # Simulate performance monitoring data
from src.monitoring.monitoring import PerformanceMonitor
import numpy as np
from dashboard import VisualizationDashboard

performance_monitor = PerformanceMonitor()
for _ in range(10):
    y_true = np.random.randint(0, 2, 100)
    y_pred = np.random.randint(0, 2, 100)
    y_prob = np.random.rand(100)
    performance_monitor.update(y_true, y_pred, y_prob, environment="production")

# Simulate drift results
drift_results = {
    "feature1": {"wasserstein_distance": 0.5},
    "feature2": {"js_divergence": 0.3},
    "feature3": {"wasserstein_distance": 0.7}
}

# Initialize and run the dashboard
dashboard = VisualizationDashboard(performance_monitor, drift_results)
dashboard.run()