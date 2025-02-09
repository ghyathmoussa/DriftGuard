import numpy as np
from src.monitoring.monitoring import PerformanceMonitor

np.random.seed(42)
y_true_train = np.random.randint(0, 2, 100)
y_pred_train = np.random.randint(0, 2, 100)
y_prob_train = np.random.rand(100)

y_true_prod = np.random.randint(0, 2, 100)
y_pred_prod = np.random.randint(0, 2, 100)
y_prob_prod = np.random.rand(100)

# Initialize performance monitor
monitor = PerformanceMonitor()

# Update with training data
for i in range(10):  # Simulate 10 batches of training data
    monitor.update(y_true_train, y_pred_train, y_prob_train, environment="training")

# Update with production data
for i in range(20):  # Simulate 20 batches of production data
    monitor.update(y_true_prod, y_pred_prod, y_prob_prod, environment="production")

# Compare training vs. production performance
comparison = monitor.compare_performance()
print("Training vs. Production Performance Comparison:")
print(comparison)

# Plot metrics over time for production
monitor.plot_metrics_over_time(environment="production")