from typing import List, Dict
import numpy as np
from transformers import AutoModel, AutoTokenizer
import torch
from scipy.spatial.distance import cosine
from collections import defaultdict

class LLMDriftDetector:
    def __init__(self, 
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2", 
                 drift_threshold: float =.15,
                reference_embeddings=None):
        """
        Initialize LLM drift detector.
        
        Args:
            model_name: Name of the model to use for embeddings
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.reference_embeddings = None
        self.drift_threshold = drift_threshold  # Configurable threshold
        
    def get_embeddings(self, texts: List[str], max_length: int = 512):
        """Generate embeddings for input texts."""
        inputs = self.tokenizer(texts, padding=True, truncation=True, 
                              return_tensors="pt", max_length=max_length)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()
    
    def set_reference(self, reference_texts: List[str]):
        """Set reference distribution using training data."""
        self.reference_embeddings = self.get_embeddings(reference_texts)
    
    def detect_drift(self, current_texts: List[str]) -> Dict:
        """
        Detect drift in current texts compared to reference.
        
        Returns:
            Dictionary containing drift metrics
        """
        if self.reference_embeddings is None:
            raise ValueError("Reference distribution not set. Call set_reference first.")
            
        current_embeddings = self.get_embeddings(current_texts)
        
        # Compute average cosine distance
        distances = [cosine(ref, curr) 
                    for ref, curr in zip(self.reference_embeddings, current_embeddings)]
        avg_distance = np.mean(distances)
        
        return {
            "drift_detected": avg_distance > self.drift_threshold,
            "drift_score": avg_distance,
            "threshold": self.drift_threshold
        }