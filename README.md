# ML Model Drift and Performance Monitoring

This project is an open-source tool designed to monitor and detect data drift, concept drift, and track the performance of machine learning models over time. It provides visualizations and alerts for anomalies, making it easier to maintain and improve the performance of deployed models.

## Overview

The tool includes the following features:

- **Data Drift Detection**: Monitors changes in input data distribution using statistical tests such as Kolmogorov-Smirnov, Chi-Square, Wasserstein distance, and Jensen-Shannon divergence.
- **Concept Drift Detection**: Detects changes in the relationship between input and output using methods like ADWIN, DDM (Drift Detection Method), and Page-Hinkley.
- **Performance Monitoring**: Tracks key metrics (accuracy, precision, recall, F1-score, ROC-AUC) over time and compares training vs. production performance.
- **Visualization**: Provides dashboards for drift and performance metrics, including time-series plots, histograms, and heatmaps.
- **Alerting**: Sends alerts via email, Slack, or other channels when drift or performance degradation is detected.
- **Storage**: Logs metrics and drift statistics for historical analysis.

## Key Features

### Data Drift Detection

- **Statistical Tests**: Uses Kolmogorov-Smirnov, Chi-Square, Wasserstein distance, and Jensen-Shannon divergence to detect drift in numerical and categorical features.
- **Feature-wise Distribution Comparison**: Compares distributions of features between training and production data.

### Concept Drift Detection

- **Model Predictions vs. Actual Labels**: Monitors model predictions and actual labels over time to detect concept drift.
- **Detection Methods**: Implements ADWIN, DDM, and Page-Hinkley methods for concept drift detection.

### Performance Monitoring

- **Metrics Tracking**: Tracks accuracy, precision, recall, F1-score, and ROC-AUC over time.
- **Training vs. Production Comparison**: Compares performance metrics between training and production environments.

### Visualization

- **Dashboards**: Provides interactive dashboards for monitoring drift and performance metrics.
- **Plots**: Includes time-series plots, histograms, and heatmaps for visualizing data and concept drift.

### Alerting

- **Email Alerts**: Sends email alerts for detected drift or performance degradation.
- **Slack Alerts**: Sends Slack messages for detected drift or performance degradation.

### Storage

- **Logging**: Logs metrics and drift statistics for historical analysis and reporting.
- **Coming Soon** Add Storage configuration (AWS, Azure, GCP)

## Installation

To install the required dependencies, run:

```sh
pip install -r requirements.txt