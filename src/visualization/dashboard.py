# visualization/dashboard.py

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from threading import Thread
import time
from alerting.alerts import AlertSystem

class VisualizationDashboard:
    def __init__(self, performance_monitor, drift_results=None, alert_system=None):
        """
        Initialize the visualization dashboard.
        
        Args:
            performance_monitor (PerformanceMonitor): Instance of the PerformanceMonitor class.
            drift_results (dict): Results from the data drift detection module.
            alert_system (AlertSystem): Instance of the AlertSystem class.
        """
        self.performance_monitor = performance_monitor
        self.drift_results = drift_results
        self.alert_system = alert_system
        self.app = dash.Dash(__name__)

        # Define layout of the dashboard
        self.app.layout = html.Div([
            html.H1("ML Model Monitoring Dashboard"),
            dcc.Tabs([
                dcc.Tab(label="Performance Metrics", children=[
                    dcc.Graph(id="performance-time-series"),
                    dcc.Graph(id="performance-comparison"),
                    dcc.Interval(id="interval-component", interval=5000, n_intervals=0)  # Update every 5 seconds
                ]),
                dcc.Tab(label="Data Drift", children=[
                    dcc.Graph(id="feature-drift-histogram"),
                    dcc.Graph(id="drift-heatmap")
                ])
            ])
        ])

        # Register callbacks
        self.app.callback(
            Output("performance-time-series", "figure"),
            Input("interval-component", "n_intervals")
        )(self.plot_performance_time_series)

        self.app.callback(
            Output("performance-comparison", "figure"),
            Input("interval-component", "n_intervals")
        )(self.plot_performance_comparison)

        self.app.callback(
            Output("feature-drift-histogram", "figure"),
            Input("interval-component", "n_intervals")
        )(self.plot_feature_drift_histogram)

        self.app.callback(
            Output("drift-heatmap", "figure"),
            Input("interval-component", "n_intervals")
        )(self.plot_drift_heatmap)

        # Start a background thread for real-time updates
        self.running = True
        self.thread = Thread(target=self.update_data, daemon=True)
        self.thread.start()

    def update_data(self):
        """
        Simulate real-time data updates.
        """
        while self.running:
            # Simulate new data
            y_true = np.random.randint(0, 2, 100)
            y_pred = np.random.randint(0, 2, 100)
            y_prob = np.random.rand(100)
            self.performance_monitor.update(y_true, y_pred, y_prob, environment="production")

            # Check for alerts
            self.check_alerts()

            # Wait for 5 seconds
            time.sleep(5)

    def check_alerts(self):
        """
        Check for performance degradation or drift and send alerts.
        """
        if self.alert_system is None:
            return

        # Example: Check for accuracy drop
        metrics_df = self.performance_monitor.get_metrics_history(environment="production")
        if len(metrics_df) == 0:
            return

        latest_accuracy = metrics_df["accuracy"].iloc[-1]
        if latest_accuracy < 0.8:  # Threshold for accuracy
            self.alert_system.send_slack_message(f"Alert: Model accuracy dropped to {latest_accuracy:.2f}!")

    def plot_performance_time_series(self, _):
        """
        Plot performance metrics over time.
        """
        metrics_df = self.performance_monitor.get_metrics_history(environment="production")
        if len(metrics_df) == 0:
            return go.Figure()

        fig = px.line(metrics_df, x="timestamp", y=metrics_df.columns[1:], title="Performance Metrics Over Time")
        return fig

    def plot_performance_comparison(self, _):
        """
        Plot a comparison of training vs. production performance.
        """
        comparison = self.performance_monitor.compare_performance()
        fig = px.bar(comparison, barmode="group", title="Training vs. Production Performance Comparison")
        return fig

    def plot_feature_drift_histogram(self, _):
        """
        Plot histograms for feature drift.
        """
        if self.drift_results is None:
            return go.Figure()

        # Example: Plot drift for the first feature
        feature = list(self.drift_results.keys())[0]
        train_data = np.random.normal(0, 1, 1000)  # Replace with actual training data
        prod_data = np.random.normal(0.5, 1, 1000)  # Replace with actual production data

        fig = go.Figure()
        fig.add_trace(go.Histogram(x=train_data, name="Training Data", opacity=0.75))
        fig.add_trace(go.Histogram(x=prod_data, name="Production Data", opacity=0.75))
        fig.update_layout(title=f"Feature Drift: {feature}", barmode="overlay")
        return fig

    def plot_drift_heatmap(self, _):
        """
        Plot a heatmap of drift scores across features.
        """
        if self.drift_results is None:
            return go.Figure()

        # Extract drift scores
        features = list(self.drift_results.keys())
        drift_scores = [result["wasserstein_distance"] if "wasserstein_distance" in result else result["js_divergence"] for result in self.drift_results.values()]

        fig = go.Figure(data=go.Heatmap(
            z=[drift_scores],
            x=features,
            y=["Drift Score"],
            colorscale="Viridis"
        ))
        fig.update_layout(title="Feature Drift Heatmap")
        return fig

    def run(self, host="0.0.0.0", port=8050):
        """
        Run the dashboard.
        """
        self.app.run_server(host=host, port=port)

    def stop(self):
        """
        Stop the background thread.
        """
        self.running = False
        self.thread.join()

