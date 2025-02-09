# Test data drift functions
import pandas as pd
import numpy as np
from src.data_drift.data_drift import detect_feature_drift, plot_feature_distributions

# Sample data
train_data = pd.DataFrame({
    "feature1": np.random.normal(0, 1, 1000),
    "feature2": np.random.choice(["A", "B", "C"], 1000)
})

prod_data = pd.DataFrame({
    "feature1": np.random.normal(0.5, 1, 1000),  # Introduce drift
    "feature2": np.random.choice(["A", "B", "C"], 1000, p=[0.1, 0.6, 0.3])  # Introduce drift
})

# Detect drift
drift_results = detect_feature_drift(train_data, prod_data)
print("Drift Results:", drift_results)

# Visualize distributions
plot_feature_distributions(train_data["feature1"], prod_data["feature1"], "feature1")