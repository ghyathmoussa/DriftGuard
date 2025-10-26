"""
LLM Drift Detection Module

This module provides drift detection capabilities for Large Language Model (LLM)
outputs using embedding-based similarity measures.
"""

from typing import List, Dict, Optional, Any
import numpy as np
from transformers import AutoModel, AutoTokenizer
import torch
from scipy.spatial.distance import cosine
from collections import defaultdict


class LLMDriftDetector:
    """
    Detector for drift in LLM outputs using embedding similarity.
    
    This class uses transformer-based embeddings to detect when the semantic
    content of LLM outputs has drifted from a reference distribution.
    
    Attributes:
        tokenizer: The tokenizer for the embedding model.
        model: The transformer model for generating embeddings.
        reference_embeddings: Reference embeddings from training/baseline data.
        drift_threshold: Threshold for detecting drift based on cosine distance.
    
    Example:
        >>> detector = LLMDriftDetector()
        >>> detector.set_reference(["reference text 1", "reference text 2"])
        >>> result = detector.detect_drift(["new text 1", "new text 2"])
        >>> print(result['drift_detected'])
    """
    
    def __init__(
        self, 
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2", 
        drift_threshold: float = 0.15,
        reference_embeddings: Optional[np.ndarray] = None
    ) -> None:
        """
        Initialize LLM drift detector.
        
        Args:
            model_name: Name of the HuggingFace model to use for embeddings.
                       Default is "sentence-transformers/all-MiniLM-L6-v2".
            drift_threshold: Threshold for drift detection. Higher values are
                           more permissive. Default is 0.15.
            reference_embeddings: Optional pre-computed reference embeddings.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.reference_embeddings = None
        self.drift_threshold = drift_threshold  # Configurable threshold
        
    def get_embeddings(self, texts: List[str], max_length: int = 512) -> np.ndarray:
        """
        Generate embeddings for input texts using the transformer model.
        
        This method tokenizes the input texts and generates dense vector
        embeddings by averaging the last hidden states of the model.
        
        Args:
            texts: List of text strings to embed.
            max_length: Maximum sequence length for tokenization. Default is 512.
        
        Returns:
            Numpy array of shape (n_texts, embedding_dim) containing the embeddings.
        
        Example:
            >>> detector = LLMDriftDetector()
            >>> embeddings = detector.get_embeddings(["Hello world", "Test text"])
            >>> embeddings.shape
            (2, 384)
        """
        inputs = self.tokenizer(texts, padding=True, truncation=True, 
                              return_tensors="pt", max_length=max_length)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()
    
    def set_reference(self, reference_texts: List[str]) -> None:
        """
        Set reference distribution using baseline/training data.
        
        This method computes and stores embeddings for the reference texts,
        which will be used as the baseline for drift detection.
        
        Args:
            reference_texts: List of reference text samples from the baseline period.
        
        Example:
            >>> detector = LLMDriftDetector()
            >>> detector.set_reference(["baseline text 1", "baseline text 2"])
        """
        self.reference_embeddings = self.get_embeddings(reference_texts)
    
    def detect_drift(self, current_texts: List[str]) -> Dict[str, Any]:
        """
        Detect drift in current texts compared to reference distribution.
        
        This method computes embeddings for the current texts and compares
        them to the reference embeddings using cosine distance. If the average
        distance exceeds the threshold, drift is detected.
        
        Args:
            current_texts: List of current text samples to check for drift.
        
        Returns:
            Dictionary containing:
                - drift_detected (bool): Whether drift was detected
                - drift_score (float): Average cosine distance from reference
                - threshold (float): The drift threshold used
        
        Raises:
            ValueError: If reference distribution has not been set.
        
        Example:
            >>> detector = LLMDriftDetector()
            >>> detector.set_reference(["ref text"])
            >>> result = detector.detect_drift(["new text"])
            >>> if result['drift_detected']:
            ...     print(f"Drift detected! Score: {result['drift_score']}")
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