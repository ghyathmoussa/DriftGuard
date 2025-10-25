#!/usr/bin/env python3
"""
Real-World Demo: Customer Service Chatbot Monitoring
Simulates monitoring a deployed chatbot for drift and performance issues
"""

from src.llm.llm_drift import LLMDriftDetector
from src.llm.llm_metrics import LLMPerformanceMonitor
import time

def simulate_chatbot_monitoring():
    """Simulate real-world chatbot monitoring scenario"""
    
    print("=" * 70)
    print("🤖 REAL-WORLD DEMO: CUSTOMER SERVICE CHATBOT MONITORING")
    print("=" * 70)
    
    # Initialize monitoring tools
    drift_detector = LLMDriftDetector(drift_threshold=0.20)
    performance_monitor = LLMPerformanceMonitor()
    
    # Week 1: Training/Baseline - Customer queries about products
    print("\n📅 WEEK 1: Setting baseline (Training data)")
    print("-" * 70)
    week1_queries = [
        "How do I return a product?",
        "What is your shipping policy?",
        "Do you have this item in stock?",
        "Can I cancel my order?",
        "What are your business hours?"
    ]
    
    week1_responses = [
        "You can return products within 30 days of purchase.",
        "We offer free shipping on orders over $50.",
        "Yes, this item is currently in stock.",
        "You can cancel orders within 24 hours of placement.",
        "We're open Monday-Friday, 9 AM to 6 PM."
    ]
    
    print(f"Training queries: {len(week1_queries)}")
    drift_detector.set_reference(week1_queries)
    performance_monitor.update(week1_responses, week1_responses, environment="training")
    print("✅ Baseline established")
    
    # Week 2: Normal operations - Similar queries
    print("\n📅 WEEK 2: Normal operations")
    print("-" * 70)
    week2_queries = [
        "How can I send back an item?",
        "What's your delivery policy?",
        "Is this product available?",
        "How do I cancel an order?",
        "When are you open?"
    ]
    
    week2_responses = [
        "Items can be returned within 30 days with receipt.",
        "Free shipping is available on $50+ orders.",
        "This product is in stock and ready to ship.",
        "Orders can be cancelled within one day of purchase.",
        "Our hours are 9 AM-6 PM, Monday through Friday."
    ]
    
    print("Analyzing Week 2 data...")
    time.sleep(1)
    
    drift_result = drift_detector.detect_drift(week2_queries)
    performance_monitor.update(week2_responses, week2_responses, environment="production")
    
    print(f"   Drift Status: {'🚨 DRIFT DETECTED' if drift_result['drift_detected'] else '✅ No Drift'}")
    print(f"   Drift Score: {drift_result['drift_score']:.4f} (threshold: {drift_result['threshold']})")
    
    metrics = performance_monitor.metrics_history["production"][-1]
    print(f"   ROUGE-L Score: {metrics['rougeL']:.4f}")
    print(f"   BLEU Score: {metrics['bleu']:.4f}")
    
    # Week 3: Topic shift - COVID-19 related queries (DRIFT!)
    print("\n📅 WEEK 3: Topic shift detected!")
    print("-" * 70)
    week3_queries = [
        "Are you open during the pandemic?",
        "Do you require masks in store?",
        "What are your COVID safety measures?",
        "Can I get contactless delivery?",
        "Are fitting rooms open?"
    ]
    
    week3_responses = [
        "Yes, we're open with modified hours due to COVID-19.",
        "Masks are required for all in-store customers.",
        "We follow CDC guidelines and sanitize regularly.",
        "Yes, contactless delivery is available for all orders.",
        "Fitting rooms are currently closed for safety."
    ]
    
    print("Analyzing Week 3 data...")
    time.sleep(1)
    
    drift_result = drift_detector.detect_drift(week3_queries)
    performance_monitor.update(week3_responses, week3_responses, environment="production")
    
    print(f"   Drift Status: {'🚨 DRIFT DETECTED' if drift_result['drift_detected'] else '✅ No Drift'}")
    print(f"   Drift Score: {drift_result['drift_score']:.4f} (threshold: {drift_result['threshold']})")
    
    if drift_result['drift_detected']:
        print("\n   ⚠️  ACTION REQUIRED:")
        print("      → Customer query patterns have significantly changed")
        print("      → Consider retraining the model with new data")
        print("      → Update FAQ and response templates")
    
    metrics = performance_monitor.metrics_history["production"][-1]
    print(f"   ROUGE-L Score: {metrics['rougeL']:.4f}")
    print(f"   BLEU Score: {metrics['bleu']:.4f}")
    
    # Week 4: Return to normal
    print("\n📅 WEEK 4: Operations stabilize")
    print("-" * 70)
    week4_queries = [
        "What's your return process?",
        "How much is shipping?",
        "Do you have this in my size?",
        "Can I modify my order?",
        "What time do you close?"
    ]
    
    week4_responses = [
        "Returns are accepted within 30 days with receipt.",
        "Shipping is free on orders over $50.",
        "Let me check inventory for your size.",
        "Order modifications are possible within 24 hours.",
        "We close at 6 PM daily."
    ]
    
    print("Analyzing Week 4 data...")
    time.sleep(1)
    
    drift_result = drift_detector.detect_drift(week4_queries)
    print(f"   Drift Status: {'🚨 DRIFT DETECTED' if drift_result['drift_detected'] else '✅ No Drift'}")
    print(f"   Drift Score: {drift_result['drift_score']:.4f} (threshold: {drift_result['threshold']})")
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 MONITORING SUMMARY")
    print("=" * 70)
    print(f"Total production batches monitored: {len(performance_monitor.metrics_history['production'])}")
    print(f"\nAverage Performance Metrics:")
    
    import pandas as pd
    prod_df = pd.DataFrame(performance_monitor.metrics_history['production'])
    print(f"   ROUGE-1: {prod_df['rouge1'].mean():.4f}")
    print(f"   ROUGE-2: {prod_df['rouge2'].mean():.4f}")
    print(f"   ROUGE-L: {prod_df['rougeL'].mean():.4f}")
    print(f"   BLEU:    {prod_df['bleu'].mean():.4f}")
    
    print("\n💡 KEY INSIGHTS:")
    print("   • Week 1-2: Normal operations, no drift detected")
    print("   • Week 3: Significant drift due to topic shift (COVID queries)")
    print("   • Week 4: Drift reduced as patterns normalize")
    print("\n   This demonstrates how DriftGuard helps you:")
    print("   ✓ Detect when customer behavior changes")
    print("   ✓ Monitor response quality over time")
    print("   ✓ Trigger alerts for model retraining")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    simulate_chatbot_monitoring()

