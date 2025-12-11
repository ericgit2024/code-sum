"""
Evaluation metrics for code summarization: BLEU, ROUGE, METEOR.
"""

import os
import json
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from typing import List, Dict
import numpy as np


# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


class EvaluationMetrics:
    """Calculates BLEU, ROUGE, and METEOR scores."""
    
    def __init__(self, config: Dict):
        """
        Initialize evaluation metrics.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )
        self.smoothing = SmoothingFunction()
        
    def calculate_bleu(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """
        Calculate BLEU scores.
        
        Args:
            reference: Reference summary
            hypothesis: Generated summary
            
        Returns:
            Dictionary with BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores
        """
        # Tokenize
        ref_tokens = nltk.word_tokenize(reference.lower())
        hyp_tokens = nltk.word_tokenize(hypothesis.lower())
        
        # Calculate BLEU scores
        bleu_1 = sentence_bleu(
            [ref_tokens], hyp_tokens,
            weights=(1, 0, 0, 0),
            smoothing_function=self.smoothing.method1
        )
        
        bleu_2 = sentence_bleu(
            [ref_tokens], hyp_tokens,
            weights=(0.5, 0.5, 0, 0),
            smoothing_function=self.smoothing.method1
        )
        
        bleu_3 = sentence_bleu(
            [ref_tokens], hyp_tokens,
            weights=(0.33, 0.33, 0.33, 0),
            smoothing_function=self.smoothing.method1
        )
        
        bleu_4 = sentence_bleu(
            [ref_tokens], hyp_tokens,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=self.smoothing.method1
        )
        
        return {
            'bleu-1': bleu_1,
            'bleu-2': bleu_2,
            'bleu-3': bleu_3,
            'bleu-4': bleu_4
        }
    
    def calculate_rouge(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """
        Calculate ROUGE scores.
        
        Args:
            reference: Reference summary
            hypothesis: Generated summary
            
        Returns:
            Dictionary with ROUGE-1, ROUGE-2, ROUGE-L F1 scores
        """
        scores = self.rouge_scorer.score(reference, hypothesis)
        
        return {
            'rouge-1': scores['rouge1'].fmeasure,
            'rouge-2': scores['rouge2'].fmeasure,
            'rouge-l': scores['rougeL'].fmeasure
        }
    
    def calculate_meteor(self, reference: str, hypothesis: str) -> float:
        """
        Calculate METEOR score.
        
        Args:
            reference: Reference summary
            hypothesis: Generated summary
            
        Returns:
            METEOR score
        """
        # Tokenize
        ref_tokens = nltk.word_tokenize(reference.lower())
        hyp_tokens = nltk.word_tokenize(hypothesis.lower())
        
        # Calculate METEOR
        score = meteor_score([ref_tokens], hyp_tokens)
        
        return score
    
    def evaluate_single(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """
        Evaluate a single prediction.
        
        Args:
            reference: Reference summary
            hypothesis: Generated summary
            
        Returns:
            Dictionary with all metric scores
        """
        results = {}
        
        # BLEU scores
        bleu_scores = self.calculate_bleu(reference, hypothesis)
        results.update(bleu_scores)
        
        # ROUGE scores
        rouge_scores = self.calculate_rouge(reference, hypothesis)
        results.update(rouge_scores)
        
        # METEOR score
        meteor = self.calculate_meteor(reference, hypothesis)
        results['meteor'] = meteor
        
        return results
    
    def evaluate_batch(self, references: List[str], hypotheses: List[str]) -> Dict[str, float]:
        """
        Evaluate a batch of predictions.
        
        Args:
            references: List of reference summaries
            hypotheses: List of generated summaries
            
        Returns:
            Dictionary with averaged metric scores
        """
        all_scores = []
        
        for ref, hyp in zip(references, hypotheses):
            scores = self.evaluate_single(ref, hyp)
            all_scores.append(scores)
        
        # Average scores
        avg_scores = {}
        for metric in all_scores[0].keys():
            avg_scores[metric] = np.mean([s[metric] for s in all_scores])
        
        return avg_scores
    
    def save_results(self, results: Dict, output_path: str):
        """
        Save evaluation results to file.
        
        Args:
            results: Evaluation results
            output_path: Path to save results
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {output_path}")
    
    def print_results(self, results: Dict):
        """
        Print evaluation results in a formatted way.
        
        Args:
            results: Evaluation results
        """
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        
        print("\nBLEU Scores:")
        for metric in ['bleu-1', 'bleu-2', 'bleu-3', 'bleu-4']:
            if metric in results:
                print(f"  {metric.upper()}: {results[metric]:.4f}")
        
        print("\nROUGE Scores:")
        for metric in ['rouge-1', 'rouge-2', 'rouge-l']:
            if metric in results:
                print(f"  {metric.upper()}: {results[metric]:.4f}")
        
        print("\nMETEOR Score:")
        if 'meteor' in results:
            print(f"  METEOR: {results['meteor']:.4f}")
        
        print("="*50 + "\n")
