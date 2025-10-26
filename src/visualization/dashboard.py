"""
Visualization Dashboard Module

This module provides a real-time web-based dashboard for monitoring ML model
performance metrics and data drift using Dash and Plotly.
"""

from typing import Dict, Optional, Any
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from threading import Thread
import time
# from src.alerting.alerts import AlertSystem


class VisualizationDashboard:
    """
    Interactive web dashboard for ML monitoring.
    
    This class creates a Dash-based web application for visualizing model
    performance metrics and data drift in real-time. It includes tabs for
    performance tracking and drift detection with auto-refresh capabilities.
    
    Attributes:
        performance_monitor: Instance of PerformanceMonitor for metrics tracking.
        drift_results: Dictionary containing drift detection results.
        alert_system: Optional AlertSystem instance for sending alerts.
        app: Dash application instance.
        running: Boolean flag controlling background updates.
        thread: Background thread for real-time data simulation.
    
    Example:
        >>> from src.monitoring.monitoring import PerformanceMonitor
        >>> monitor = PerformanceMonitor()
        >>> dashboard = VisualizationDashboard(monitor)
        >>> dashboard.run(host="localhost", port=8050)
    """
    
    def __init__(
        self,
        performance_monitor: Any,
        drift_results: Optional[Dict[str, Any]] = None,
        alert_system: Optional[Any] = None
    ) -> None:
        """
        Initialize the visualization dashboard.
        
        Args:
            performance_monitor: Instance of the PerformanceMonitor class.
            drift_results: Results from the data drift detection module (optional).
            alert_system: Instance of the AlertSystem class for alerts (optional).
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

    def update_data(self) -> None:
        """
        Simulate real-time data updates in a background thread.
        
        This method continuously generates simulated monitoring data and updates
        the performance monitor. It runs until the 'running' flag is set to False.
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

    def check_alerts(self) -> None:
        """
        Check for performance degradation or drift and send alerts.
        
        Monitors the latest metrics and sends alerts via the configured alert
        system if thresholds are exceeded (e.g., accuracy drops below 0.8).
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

    def plot_performance_time_series(self, _: int) -> go.Figure:
        """
        Plot performance metrics over time as a line chart.
        
        Args:
            _: Unused parameter (n_intervals from Dash callback).
        
        Returns:
            Plotly Figure object containing the time series plot.
        """
        metrics_df = self.performance_monitor.get_metrics_history(environment="production")
        if len(metrics_df) == 0:
            return go.Figure()

        fig = px.line(metrics_df, x="timestamp", y=metrics_df.columns[1:], title="Performance Metrics Over Time")
        return fig

    def plot_performance_comparison(self, _: int) -> go.Figure:
        """
        Plot a comparison of training vs. production performance.
        
        Args:
            _: Unused parameter (n_intervals from Dash callback).
        
        Returns:
            Plotly Figure object containing the bar chart comparison.
        """
        comparison = self.performance_monitor.compare_performance()
        fig = px.bar(comparison, barmode="group", title="Training vs. Production Performance Comparison")
        return fig

    def plot_feature_drift_histogram(self, _: int) -> go.Figure:
        """
        Plot histograms for feature drift visualization.
        
        Args:
            _: Unused parameter (n_intervals from Dash callback).
        
        Returns:
            Plotly Figure object containing overlaid histograms.
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

    def plot_drift_heatmap(self, _: int) -> go.Figure:
        """
        Plot a heatmap of drift scores across features.
        
        Args:
            _: Unused parameter (n_intervals from Dash callback).
        
        Returns:
            Plotly Figure object containing the drift scores heatmap.
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

    def run(self, host: str = "0.0.0.0", port: int = 8050) -> None:
        """
        Run the dashboard web server.
        
        Starts the Dash application server, making the dashboard accessible
        via web browser at the specified host and port.
        
        Args:
            host: Host address to bind the server to. Default is "0.0.0.0".
            port: Port number to run the server on. Default is 8050.
        
        Example:
            >>> dashboard = VisualizationDashboard(monitor)
            >>> dashboard.run(host="localhost", port=8050)
        """
        self.app.run_server(host=host, port=port)

    def stop(self) -> None:
        """
        Stop the background thread and clean up resources.
        
        Sets the running flag to False and waits for the background
        data simulation thread to terminate gracefully.
        """
        self.running = False
        self.thread.join()

