# Drift Detection

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import seaborn as sns
from utils.logger import log_drift

# For Numerical Data

from scipy.stats import ks_2samp

def detect_numerical_drift(train_data, prod_data, feature, threshold=0.05):
    """
    Detect drift in numerical features using the KS test.
    
    Args:
        train_data (pd.Series): Training data for the feature.
        prod_data (pd.Series): Production data for the feature.
        feature (str): Name of the feature being analyzed.
        threshold (float): Significance level for drift detection.
    
    Returns:
        bool: True if drift is detected, False otherwise.
    """
    stat, p_value = ks_2samp(train_data, prod_data)
    if p_value < threshold:
        print(f"Drift detected in feature: {feature} (p-value: {p_value})")
        print(f"KS Statistic: {stat}")
        return True, p_value, stat
    else:
        print(f"No drift detected in feature: {feature} (p-value: {p_value})")
        print(f"KS Statistic: {stat}")
        return False
    
# For categorical data

def detect_categorical_drift(train_data, prod_data, feature, threshold=0.05):
    """
    Detect drift in categorical features using the Chi-Square test.
    
    Args:
        train_data (pd.Series): Training data for the feature.
        prod_data (pd.Series): Production data for the feature.
        threshold (float): Significance level for drift detection.
    
    Returns:
        bool: True if drift is detected, False otherwise.
    """
    # Create contingency table
    train_counts = train_data.value_counts().sort_index()
    prod_counts = prod_data.value_counts().sort_index()
    contingency_table = pd.concat([train_counts, prod_counts], axis=1).fillna(0).values
    
    # Perform Chi-Square test
    _, p_value, _, _ = chi2_contingency(contingency_table)
    if p_value < threshold:
        print(f"Drift detected in feature: {feature} (p-value: {p_value})")
        return True, p_value
    else:
        print(f"No drift detected in feature: {feature} (p-value: {p_value})")
        return False

def compute_wasserstein_distance(train_data, prod_data):
    """
    Compute Wasserstein distance between two distributions.
    
    Args:
        train_data (pd.Series): Training data for the feature.
        prod_data (pd.Series): Production data for the feature.
    
    Returns:
        float: Wasserstein distance.
    """
    return wasserstein_distance(train_data, prod_data)

def compute_js_divergence(train_data, prod_data):
    """
    Compute Jensen-Shannon divergence between two distributions.
    
    Args:
        train_data (pd.Series): Training data for the feature.
        prod_data (pd.Series): Production data for the feature.
    
    Returns:
        float: Jensen-Shannon divergence.
    """
    # Normalize the distributions
    train_dist = train_data.value_counts(normalize=True).sort_index()
    prod_dist = prod_data.value_counts(normalize=True).sort_index()
    
    # Align indices and fill missing values with 0
    aligned_dist = pd.concat([train_dist, prod_dist], axis=1).fillna(0)
    
    # Compute JS divergence
    return jensenshannon(aligned_dist.iloc[:, 0], aligned_dist.iloc[:, 1])

def detect_feature_drift(train_data, prod_data, threshold=0.05):
    """
    Detect drift for all features in a dataset.
    
    Args:
        train_data (pd.DataFrame): Training data.
        prod_data (pd.DataFrame): Production data.
        threshold (float): Significance level for drift detection.
    
    Returns:
        dict: Dictionary with drift results for each feature.
    """
    drift_results = {}
    
    for feature in train_data.columns:
        if np.issubdtype(train_data[feature].dtype, np.number):  # Numerical feature
            drift_results[feature] = {
                "drift_detected": detect_numerical_drift(train_data[feature], prod_data[feature], threshold),
                "wasserstein_distance": compute_wasserstein_distance(train_data[feature], prod_data[feature])
            }
        else:  # Categorical feature
            drift_results[feature] = {
                "drift_detected": detect_categorical_drift(train_data[feature], prod_data[feature], threshold),
                "js_divergence": compute_js_divergence(train_data[feature], prod_data[feature])
            }
    log_drift(drift_results)
    return drift_results

# Visualizing drift

def plot_feature_distributions(train_data, prod_data, feature):
    """
    Plot distributions of a feature in training and production data.
    
    Args:
        train_data (pd.Series): Training data for the feature.
        prod_data (pd.Series): Production data for the feature.
        feature (str): Name of the feature.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(train_data, label="Training Data", kde=True, color="blue")
    sns.histplot(prod_data, label="Production Data", kde=True, color="red")
    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Density")
    plt.legend()
    plt.show()


def wasserstein_distance_metric(train_data, prod_data):
    """
    Compute Wasserstein distance between two distributions.
    
    Args:
        train_data (pd.Series): Training data for the feature.
        prod_data (pd.Series): Production data for the feature.
    
    Returns:
        float: Wasserstein distance.
    """
    return wasserstein_distance(train_data, prod_data)

