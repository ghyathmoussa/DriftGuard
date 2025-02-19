import unittest
from src.llm.llm_metrics import LLMPerformanceMonitor
from src.llm.llm_drift import LLMDriftDetector

class TestLLMMonitoring(unittest.TestCase):
    def setUp(self):
        self.llm_monitor = LLMPerformanceMonitor()
        self.drift_detector = LLMDriftDetector()
        
        # Test data
        self.reference_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Python is a popular programming language.",
            "Machine learning models need monitoring."
        ]
        
        self.generated_texts = [
            "A swift brown fox leaps across the sleeping dog.",
            "Python is widely used in software development.",
            "ML models require continuous monitoring."
        ]

    def test_performance_metrics(self):
        """Test if performance metrics are computed correctly"""
        self.llm_monitor.update(self.reference_texts, self.generated_texts)
        metrics = self.llm_monitor.metrics_history["production"][-1]
        
        # Check if all expected metrics are present
        expected_metrics = ["rouge1", "rouge2", "rougeL", "bleu", "bertscore_f1"]
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], float)
            self.assertTrue(0 <= metrics[metric] <= 1)

    def test_drift_detection(self):
        """Test if drift detection works"""
        self.drift_detector.set_reference(self.reference_texts)
        drift_results = self.drift_detector.detect_drift(self.generated_texts)
        
        self.assertIn("drift_detected", drift_results)
        self.assertIn("drift_score", drift_results)
        self.assertIsInstance(drift_results["drift_score"], float)

if __name__ == '__main__':
    unittest.main() 