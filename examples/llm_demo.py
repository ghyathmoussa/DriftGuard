from src.llm.llm_metrics import LLMPerformanceMonitor
from src.llm.llm_drift import LLMDriftDetector
import time

def run_demo():
    # Initialize monitors
    llm_monitor = LLMPerformanceMonitor()
    drift_detector = LLMDriftDetector()
    
    # Example data - training distribution
    reference_texts = [
        "The weather is sunny today in San Francisco.",
        "Machine learning models need regular monitoring.",
        "The stock market showed positive trends this week.",
        "Scientists discovered a new species in the Amazon.",
        "Electric vehicles are becoming more popular."
    ]
    
    # Example data - production distribution (slight drift)
    production_texts = [
        "Today's weather in San Francisco is cloudy.",
        "AI models require continuous performance tracking.",
        "Stock markets demonstrated bearish patterns.",
        "Researchers found an unknown species in rainforest.",
        "EVs are gaining market share globally."
    ]
    
    print("Starting LLM monitoring demo...")
    print("\n1. Testing Performance Monitoring:")
    print("----------------------------------")
    
    # Monitor performance
    start_time = time.time()
    llm_monitor.update(reference_texts, production_texts)
    metrics = llm_monitor.metrics_history["production"][-1]
    
    print(f"Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    print(f"\nTime taken: {time.time() - start_time:.2f} seconds")
    
    print("\n2. Testing Drift Detection:")
    print("---------------------------")
    
    # Monitor drift
    start_time = time.time()
    drift_detector.set_reference(reference_texts)
    drift_results = drift_detector.detect_drift(production_texts)
    
    print(f"Drift Results:")
    for key, value in drift_results.items():
        if isinstance(value, bool):
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value:.4f}")
    print(f"\nTime taken: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    run_demo() 