#!/usr/bin/env python3
"""
Comprehensive DriftGuard Demo
Demonstrates: Data Drift, Concept Drift, Performance Monitoring
"""

import numpy as np
import pandas as pd
from src.data_drift.data_drift import detect_numerical_drift, detect_categorical_drift
from src.concept_drift.drift_detection import ConceptDriftDetector
from src.monitoring.monitoring import PerformanceMonitor

def demo_data_drift():
    """Demo: Detect data drift in features"""
    print("\n" + "=" * 60)
    print("DEMO 1: DATA DRIFT DETECTION")
    print("=" * 60)
    
    # Create training data
    np.random.seed(42)
    train_age = np.random.normal(35, 10, 1000)
    train_income = np.random.normal(50000, 15000, 1000)
    
    # Create production data with drift
    prod_age = np.random.normal(40, 12, 1000)  # Age shifted
    prod_income = np.random.normal(50000, 15000, 1000)  # No drift
    
    print("\n1. Testing NUMERICAL drift (Age feature):")
    drift_result, p_value, stat = detect_numerical_drift(
        train_age, prod_age, "age", threshold=0.05
    )
    
    print("\n2. Testing NUMERICAL drift (Income feature):")
    detect_numerical_drift(
        train_income, prod_income, "income", threshold=0.05
    )
    
    # Categorical drift
    print("\n3. Testing CATEGORICAL drift:")
    train_category = pd.Series(np.random.choice(['A', 'B', 'C'], 1000, p=[0.5, 0.3, 0.2]))
    prod_category = pd.Series(np.random.choice(['A', 'B', 'C'], 1000, p=[0.3, 0.4, 0.3]))
    detect_categorical_drift(train_category, prod_category, "category", threshold=0.05)


def demo_concept_drift():
    """Demo: Detect concept drift in model predictions"""
    print("\n" + "=" * 60)
    print("DEMO 2: CONCEPT DRIFT DETECTION")
    print("=" * 60)
    
    # Simulate a scenario where model performance degrades
    np.random.seed(42)
    
    # First 300 predictions: Good performance
    y_true_good = np.random.binomial(1, 0.7, 300)
    y_pred_good = np.random.binomial(1, 0.65, 300)
    
    # Next 200 predictions: Performance degrades (concept drift)
    y_true_drift = np.random.binomial(1, 0.3, 200)
    y_pred_drift = np.random.binomial(1, 0.65, 200)
    
    # Combine
    y_true = np.concatenate([y_true_good, y_true_drift])
    y_pred = np.concatenate([y_pred_good, y_pred_drift])
    
    print("\nMonitoring predictions with ADWIN detector...")
    detector = ConceptDriftDetector(method="ADWIN")
    
    drift_points = []
    for i, (true, pred) in enumerate(zip(y_true, y_pred)):
        drift_detected = detector.update(true, pred)
        if drift_detected:
            drift_points.append(i)
            print(f"   🚨 DRIFT DETECTED at sample {i}")
    
    print(f"\nTotal drift points detected: {len(drift_points)}")


def demo_performance_monitoring():
    """Demo: Track model performance over time"""
    print("\n" + "=" * 60)
    print("DEMO 3: PERFORMANCE MONITORING")
    print("=" * 60)
    
    # Initialize monitor
    monitor = PerformanceMonitor()
    
    np.random.seed(42)
    
    # Simulate training performance
    print("\n1. Simulating TRAINING performance...")
    for batch in range(3):
        y_true = np.random.binomial(1, 0.6, 200)
        y_pred = np.random.binomial(1, 0.6, 200)
        y_prob = np.random.rand(200)
        monitor.update(y_true, y_pred, y_prob, environment="training")
    
    # Simulate production performance (slightly worse)
    print("2. Simulating PRODUCTION performance...")
    for batch in range(3):
        y_true = np.random.binomial(1, 0.6, 200)
        y_pred = np.random.binomial(1, 0.5, 200)  # Worse predictions
        y_prob = np.random.rand(200)
        monitor.update(y_true, y_pred, y_prob, environment="production")
    
    # Compare performance
    print("\n3. Performance Comparison:")
    print("-" * 60)
    comparison = monitor.compare_performance()
    print(comparison)
    
    # Get latest production metrics
    print("\n4. Latest Production Metrics:")
    print("-" * 60)
    prod_history = monitor.get_metrics_history("production")
    latest = prod_history.iloc[-1]
    print(f"   Accuracy:  {latest['accuracy']:.4f}")
    print(f"   Precision: {latest['precision']:.4f}")
    print(f"   Recall:    {latest['recall']:.4f}")
    print(f"   F1 Score:  {latest['f1_score']:.4f}")


def main():
    """Run all demos"""
    print("\n" + "=" * 60)
    print("🚀 DRIFTGUARD COMPREHENSIVE DEMO")
    print("=" * 60)
    print("\nThis demo showcases DriftGuard's main capabilities:")
    print("  • Data Drift Detection (statistical tests)")
    print("  • Concept Drift Detection (ADWIN)")
    print("  • Performance Monitoring (metrics tracking)")
    
    try:
        demo_data_drift()
        demo_concept_drift()
        demo_performance_monitoring()
        
        print("\n" + "=" * 60)
        print("✅ ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nNext steps:")
        print("  • Run 'python examples/llm_demo.py' for LLM-specific features")
        print("  • Check 'logs/metrics_drift.log' for logged results")
        print("  • Explore visualization with dashboard.py")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

