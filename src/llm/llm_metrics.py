from typing import Dict, List, Optional
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import evaluate


class LLMPerformanceMonitor:
    def __init__(self):
        """Initialize LLM performance monitoring."""
        self.metrics_history = {
            "training": [],
            "production": []
        }
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.perplexity = evaluate.load("perplexity", module_type="metric")
        self.bertscore = evaluate.load("bertscore")

    def compute_metrics(self, 
                       references: List[str], 
                       predictions: List[str],
                       input_texts: Optional[List[str]] = None) -> Dict:
        """
        Compute LLM-specific metrics.
        
        Args:
            references: List of reference/ground truth texts
            predictions: List of model generated texts
            input_texts: Optional list of input prompts
            
        Returns:
            Dictionary of computed metrics
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

    def update(self, 
               references: List[str], 
               predictions: List[str],
               input_texts: Optional[List[str]] = None, 
               environment: str = "production"):
        """
        Update metrics history with new data.
        """
        metrics = self.compute_metrics(references, predictions, input_texts)
        self.metrics_history[environment].append(metrics) 