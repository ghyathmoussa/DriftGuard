"""
Data Drift Detection Module

This module provides functions for detecting data drift between training and production
data distributions using various statistical tests and distance metrics.
"""

from typing import Dict, Tuple, Union, Optional, Any
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.logger import log_drift


def detect_numerical_drift(
    train_data: pd.Series,
    prod_data: pd.Series,
    feature: str,
    threshold: float = 0.05
) -> Union[Tuple[bool, float, float], bool]:
    """
    Detect drift in numerical features using the Kolmogorov-Smirnov (KS) test.
    
    The KS test is a non-parametric test that compares two distributions
    and returns a p-value indicating whether they are significantly different.
    
    Args:
        train_data: Training data for the feature.
        prod_data: Production data for the feature.
        feature: Name of the feature being analyzed.
        threshold: Significance level for drift detection (default: 0.05).
    
    Returns:
        If drift is detected: Tuple of (True, p_value, ks_statistic)
        If no drift detected: False
    
    Example:
        >>> train = pd.Series([1, 2, 3, 4, 5])
        >>> prod = pd.Series([10, 20, 30, 40, 50])
        >>> detect_numerical_drift(train, prod, "age", threshold=0.05)
        (True, 0.001, 0.95)
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


def detect_categorical_drift(
    train_data: pd.Series,
    prod_data: pd.Series,
    feature: str,
    threshold: float = 0.05
) -> Union[Tuple[bool, float], bool]:
    """
    Detect drift in categorical features using the Chi-Square test.
    
    The Chi-Square test evaluates whether the distribution of categorical
    values has changed significantly between training and production data.
    
    Args:
        train_data: Training data for the feature.
        prod_data: Production data for the feature.
        feature: Name of the feature being analyzed.
        threshold: Significance level for drift detection (default: 0.05).
    
    Returns:
        If drift is detected: Tuple of (True, p_value)
        If no drift detected: False
    
    Example:
        >>> train = pd.Series(['A', 'B', 'A', 'C'])
        >>> prod = pd.Series(['B', 'B', 'C', 'C'])
        >>> detect_categorical_drift(train, prod, "category", threshold=0.05)
        (True, 0.02)
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

def compute_wasserstein_distance(
    train_data: pd.Series,
    prod_data: pd.Series
) -> float:
    """
    Compute Wasserstein distance (Earth Mover's Distance) between two distributions.
    
    The Wasserstein distance measures the minimum amount of work needed to transform
    one distribution into another. Lower values indicate more similar distributions.
    
    Args:
        train_data: Training data for the feature.
        prod_data: Production data for the feature.
    
    Returns:
        Wasserstein distance between the two distributions.
    
    Example:
        >>> train = pd.Series([1, 2, 3, 4, 5])
        >>> prod = pd.Series([2, 3, 4, 5, 6])
        >>> compute_wasserstein_distance(train, prod)
        1.0
    """
    return wasserstein_distance(train_data, prod_data)


def compute_js_divergence(
    train_data: pd.Series,
    prod_data: pd.Series
) -> float:
    """
    Compute Jensen-Shannon divergence between two distributions.
    
    The Jensen-Shannon divergence is a symmetric measure of similarity between
    two probability distributions. It ranges from 0 (identical) to 1 (completely different).
    
    Args:
        train_data: Training data for the feature.
        prod_data: Production data for the feature.
    
    Returns:
        Jensen-Shannon divergence between the two distributions.
    
    Example:
        >>> train = pd.Series(['A', 'B', 'A', 'C'])
        >>> prod = pd.Series(['A', 'A', 'B', 'C'])
        >>> compute_js_divergence(train, prod)
        0.15
    """
    # Normalize the distributions
    train_dist = train_data.value_counts(normalize=True).sort_index()
    prod_dist = prod_data.value_counts(normalize=True).sort_index()
    
    # Align indices and fill missing values with 0
    aligned_dist = pd.concat([train_dist, prod_dist], axis=1).fillna(0)
    
    # Compute JS divergence
    return jensenshannon(aligned_dist.iloc[:, 0], aligned_dist.iloc[:, 1])

def detect_feature_drift(
    train_data: pd.DataFrame,
    prod_data: pd.DataFrame,
    threshold: float = 0.05
) -> Dict[str, Dict[str, Any]]:
    """
    Detect drift for all features in a dataset.
    
    This function automatically detects the data type of each feature and applies
    the appropriate drift detection method (KS test for numerical, Chi-Square for categorical).
    
    Args:
        train_data: Training dataset containing all features.
        prod_data: Production dataset containing all features.
        threshold: Significance level for drift detection (default: 0.05).
    
    Returns:
        Dictionary mapping feature names to their drift detection results.
        Each result contains 'drift_detected' and either 'wasserstein_distance' 
        (numerical) or 'js_divergence' (categorical).
    
    Example:
        >>> train = pd.DataFrame({'age': [25, 30, 35], 'city': ['A', 'B', 'A']})
        >>> prod = pd.DataFrame({'age': [50, 55, 60], 'city': ['C', 'C', 'B']})
        >>> results = detect_feature_drift(train, prod)
        >>> results['age']['drift_detected']
        True
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


def plot_feature_distributions(
    train_data: pd.Series,
    prod_data: pd.Series,
    feature: str
) -> None:
    """
    Plot distributions of a feature in training and production data.
    
    Creates a histogram with KDE overlay comparing the distributions of
    the same feature in training and production datasets.
    
    Args:
        train_data: Training data for the feature.
        prod_data: Production data for the feature.
        feature: Name of the feature to be displayed in the plot title.
    
    Returns:
        None. Displays the plot using matplotlib.
    
    Example:
        >>> train = pd.Series([1, 2, 3, 4, 5], name='age')
        >>> prod = pd.Series([6, 7, 8, 9, 10], name='age')
        >>> plot_feature_distributions(train, prod, 'age')
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(train_data, label="Training Data", kde=True, color="blue")
    sns.histplot(prod_data, label="Production Data", kde=True, color="red")
    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Density")
    plt.legend()
    plt.show()


def wasserstein_distance_metric(
    train_data: pd.Series,
    prod_data: pd.Series
) -> float:
    """
    Compute Wasserstein distance between two distributions.
    
    This is an alias for compute_wasserstein_distance() for backward compatibility.
    
    Args:
        train_data: Training data for the feature.
        prod_data: Production data for the feature.
    
    Returns:
        Wasserstein distance between the two distributions.
    
    See Also:
        compute_wasserstein_distance: The main implementation of this metric.
    """
    return wasserstein_distance(train_data, prod_data)

