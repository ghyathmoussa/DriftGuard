#!/usr/bin/env python3
"""
Simple DriftGuard Demo - LLM Drift Detection
"""

from src.llm.llm_drift import LLMDriftDetector

def simple_llm_demo():
    print("=" * 60)
    print("DriftGuard - Simple LLM Demo")
    print("=" * 60)
    
    # Initialize detector
    print("\n1. Initializing LLM Drift Detector...")
    drift_detector = LLMDriftDetector(drift_threshold=0.15)
    
    # Reference data (training/baseline)
    reference_texts = [
        "What is machine learning?",
        "How do neural networks work?",
        "Explain natural language processing.",
        "What are transformers in AI?",
        "Define deep learning."
    ]
    
    print("\n2. Setting reference baseline...")
    print(f"   Reference texts: {len(reference_texts)} samples")
    drift_detector.set_reference(reference_texts)
    
    # Test Case 1: Similar data (No Drift Expected)
    print("\n3. Testing SIMILAR data (No drift expected)...")
    similar_texts = [
        "Can you explain machine learning?",
        "How do artificial neural networks function?",
        "Please explain NLP.",
        "What is a transformer model?",
        "Can you define deep learning?"
    ]
    
    result1 = drift_detector.detect_drift(similar_texts)
    print(f"   Drift Detected: {result1['drift_detected']}")
    print(f"   Drift Score: {result1['drift_score']:.4f}")
    print(f"   Threshold: {result1['threshold']}")
    
    # Test Case 2: Different domain (Drift Expected)
    print("\n4. Testing DIFFERENT data (Drift expected)...")
    different_texts = [
        "What's the weather like today?",
        "How do I cook pasta?",
        "Where is the nearest coffee shop?",
        "What time does the movie start?",
        "Can you recommend a good book?"
    ]
    
    result2 = drift_detector.detect_drift(different_texts)
    print(f"   Drift Detected: {result2['drift_detected']}")
    print(f"   Drift Score: {result2['drift_score']:.4f}")
    print(f"   Threshold: {result2['threshold']}")
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)

if __name__ == "__main__":
    simple_llm_demo()

