"""
LLM Performance Monitoring Module

This module provides comprehensive performance monitoring for Large Language Models
using various NLP metrics including BLEU, ROUGE, BERTScore, and Perplexity.
"""

from typing import Dict, List, Optional, Any
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import evaluate


class LLMPerformanceMonitor:
    """
    Monitor and track LLM performance metrics over time.
    
    This class computes various metrics for evaluating LLM outputs including
    BLEU, ROUGE, BERTScore, and Perplexity. It maintains separate histories
    for training and production environments.
    
    Attributes:
        metrics_history: Dictionary storing metrics for training and production.
        rouge_scorer: ROUGE metric scorer.
        perplexity: Perplexity metric evaluator.
        bertscore: BERTScore metric evaluator.
    
    Example:
        >>> monitor = LLMPerformanceMonitor()
        >>> references = ["The cat sat on the mat"]
        >>> predictions = ["A cat was sitting on the mat"]
        >>> metrics = monitor.compute_metrics(references, predictions)
        >>> print(metrics['bleu'])
    """
    
    def __init__(self) -> None:
        """
        Initialize LLM performance monitoring.
        
        Sets up metric scorers and initializes empty history for both
        training and production environments.
        """
        self.metrics_history = {
            "training": [],
            "production": []
        }
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.perplexity = evaluate.load("perplexity", module_type="metric")
        self.bertscore = evaluate.load("bertscore")

    def compute_metrics(
        self, 
        references: List[str], 
        predictions: List[str],
        input_texts: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Compute comprehensive LLM-specific metrics.
        
        This method calculates multiple quality metrics for LLM outputs including:
        - ROUGE scores (overlap-based metrics)
        - BLEU score (n-gram precision)
        - BERTScore (semantic similarity)
        - Perplexity (if input texts provided)
        
        Args:
            references: List of reference/ground truth texts.
            predictions: List of model generated texts.
            input_texts: Optional list of input prompts for perplexity calculation.
            
        Returns:
            Dictionary containing computed metrics with keys:
                - rouge1, rouge2, rougeL: ROUGE F1 scores
                - bleu: BLEU score
                - bertscore_f1: BERTScore F1
                - perplexity: Perplexity (if input_texts provided)
        
        Example:
            >>> monitor = LLMPerformanceMonitor()
            >>> refs = ["The quick brown fox"]
            >>> preds = ["The fast brown fox"]
            >>> metrics = monitor.compute_metrics(refs, preds)
            >>> print(f"BLEU: {metrics['bleu']:.3f}")
        """
        metrics = {}
        
        # ROUGE scores
        rouge_scores = [self.rouge_scorer.score(ref, pred) 
                       for ref, pred in zip(references, predictions)]
        
        metrics["rouge1"] = np.mean([score["rouge1"].fmeasure for score in rouge_scores])
        metrics["rouge2"] = np.mean([score["rouge2"].fmeasure for score in rouge_scores])
        metrics["rougeL"] = np.mean([score["rougeL"].fmeasure for score in rouge_scores])
        
        # BLEU score
        metrics["bleu"] = np.mean([sentence_bleu([ref.split()], pred.split()) 
                                 for ref, pred in zip(references, predictions)])
        
        # BERTScore
        bert_scores = self.bertscore.compute(predictions=predictions, 
                                           references=references, 
                                           lang="en")
        metrics["bertscore_f1"] = np.mean(bert_scores["f1"])
        
        # Perplexity (if input texts are provided)
        if input_texts:
            perplexity_scores = self.perplexity.compute(predictions=predictions,
                                                       model_id="gpt2")
            metrics["perplexity"] = np.mean(perplexity_scores["perplexities"])
            
        return metrics

    def update(
        self, 
        references: List[str], 
        predictions: List[str],
        input_texts: Optional[List[str]] = None, 
        environment: str = "production"
    ) -> None:
        """
        Update metrics history with new data.
        
        This method computes metrics for the provided data and stores them
        in the history for the specified environment.
        
        Args:
            references: List of reference/ground truth texts.
            predictions: List of model generated texts.
            input_texts: Optional list of input prompts.
            environment: Environment identifier ("training" or "production").
                        Default is "production".
        
        Example:
            >>> monitor = LLMPerformanceMonitor()
            >>> monitor.update(
            ...     references=["ground truth"],
            ...     predictions=["model output"],
            ...     environment="production"
            ... )
        """
        metrics = self.compute_metrics(references, predictions, input_texts)
        self.metrics_history[environment].append(metrics) 