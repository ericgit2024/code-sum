"""
Uncertainty-Aware Agent using Monte Carlo Dropout for confidence estimation.

This module implements Option 6: Uncertainty-Aware Summarization by:
1. Enabling dropout during inference (Monte Carlo Dropout)
2. Generating multiple summaries with different dropout patterns
3. Calculating variance across predictions to estimate uncertainty
4. Selectively refining low-confidence parts
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
from collections import Counter


class UncertaintyAgent:
    """Agent that quantifies uncertainty in generated summaries using Monte Carlo Dropout."""
    
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, 
                 config: Dict, n_samples: int = 5):
        """
        Initialize uncertainty agent.
        
        Args:
            model: Language model
            tokenizer: Tokenizer
            config: Configuration dictionary
            n_samples: Number of Monte Carlo samples (default: 5)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.n_samples = n_samples
        self.confidence_threshold = config.get('uncertainty_agent', {}).get('confidence_threshold', 0.6)
        self.max_refinement_iterations = config.get('uncertainty_agent', {}).get('max_refinement_iterations', 2)
        
        print(f"[UncertaintyAgent] Initialized with n_samples={n_samples}, threshold={self.confidence_threshold}")
    
    def enable_dropout(self):
        """
        Enable dropout layers during inference for Monte Carlo sampling.
        
        This is the key technique: normally dropout is disabled during inference,
        but we keep it active to get different predictions each time.
        """
        for module in self.model.modules():
            if module.__class__.__name__.startswith('Dropout'):
                module.train()  # Set to training mode to enable dropout
    
    def disable_dropout(self):
        """Disable dropout layers (restore normal inference mode)."""
        for module in self.model.modules():
            if module.__class__.__name__.startswith('Dropout'):
                module.eval()
    
    def generate_multiple_summaries(self, prompt: str, n_samples: int = None) -> List[str]:
        """
        Generate multiple summaries using Monte Carlo Dropout.
        
        Args:
            prompt: Input prompt
            n_samples: Number of samples (uses self.n_samples if None)
            
        Returns:
            List of generated summaries
        """
        if n_samples is None:
            n_samples = self.n_samples
        
        summaries = []
        
        # Enable dropout for uncertainty estimation
        self.enable_dropout()
        
        try:
            for i in range(n_samples):
                # Tokenize
                inputs = self.tokenizer(
                    prompt,
                    return_tensors='pt',
                    truncation=True,
                    max_length=512
                ).to(self.model.device)
                
                # Generate with different dropout patterns
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=150,
                        min_new_tokens=10,
                        temperature=0.7,  # Some randomness for diversity
                        do_sample=True,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                
                # Decode
                full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract summary (after the prompt)
                summary = self._extract_summary(full_text, prompt)
                summaries.append(summary)
        
        finally:
            # Always restore normal inference mode
            self.disable_dropout()
        
        return summaries
    
    def _extract_summary(self, full_text: str, prompt: str) -> str:
        """
        Extract generated summary from full text.
        
        Args:
            full_text: Complete generated text
            prompt: Original prompt
            
        Returns:
            Extracted summary
        """
        # Remove prompt from output
        if prompt in full_text:
            summary = full_text.split(prompt)[-1].strip()
        else:
            summary = full_text.strip()
        
        # Clean up common artifacts
        summary = re.sub(r'^(Summary:|Docstring:)\s*', '', summary, flags=re.IGNORECASE)
        summary = re.sub(r'```.*?```', '', summary, flags=re.DOTALL)  # Remove code blocks
        summary = summary.split('\n\n')[0]  # Take first paragraph
        
        return summary.strip()
    
    def calculate_sentence_variance(self, summaries: List[str]) -> Tuple[List[str], List[float]]:
        """
        Calculate variance across generated summaries at sentence level.
        
        Args:
            summaries: List of generated summaries
            
        Returns:
            Tuple of (sentences, confidence_scores)
        """
        # Split all summaries into sentences
        all_sentences = [self._split_sentences(s) for s in summaries]
        
        # Find the most common structure (number of sentences)
        sentence_counts = [len(s) for s in all_sentences]
        most_common_count = Counter(sentence_counts).most_common(1)[0][0]
        
        # Filter to summaries with the most common structure
        filtered_summaries = [s for s in all_sentences if len(s) == most_common_count]
        
        if not filtered_summaries or most_common_count == 0:
            # Fallback: return first summary with low confidence
            return self._split_sentences(summaries[0]), [0.5] * len(self._split_sentences(summaries[0]))
        
        # Calculate confidence for each sentence position
        confidence_scores = []
        consensus_sentences = []
        
        for i in range(most_common_count):
            # Get all variants of sentence i
            sentence_variants = [s[i] for s in filtered_summaries]
            
            # Calculate similarity (simple: count exact matches)
            most_common_sentence = Counter(sentence_variants).most_common(1)[0]
            agreement_ratio = most_common_sentence[1] / len(sentence_variants)
            
            # Confidence = agreement ratio (0.0 to 1.0)
            confidence_scores.append(agreement_ratio)
            consensus_sentences.append(most_common_sentence[0])
        
        return consensus_sentences, confidence_scores
    
    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting (can be improved with NLTK)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def compute_confidence_scores(self, summaries: List[str]) -> Dict:
        """
        Compute confidence scores from multiple summaries.
        
        Args:
            summaries: List of generated summaries
            
        Returns:
            Dictionary with confidence metrics
        """
        sentences, confidence_scores = self.calculate_sentence_variance(summaries)
        
        # Calculate aggregate metrics
        mean_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        min_confidence = np.min(confidence_scores) if confidence_scores else 0.0
        
        # Identify low-confidence sentence indices
        low_confidence_indices = [
            i for i, score in enumerate(confidence_scores) 
            if score < self.confidence_threshold
        ]
        
        return {
            'sentences': sentences,
            'confidence_scores': confidence_scores,
            'mean_confidence': float(mean_confidence),
            'min_confidence': float(min_confidence),
            'low_confidence_indices': low_confidence_indices,
            'n_samples': len(summaries)
        }
    
    def refine_uncertain_parts(self, code: str, sentences: List[str], 
                              low_confidence_indices: List[int]) -> str:
        """
        Selectively refine low-confidence sentences.
        
        Args:
            code: Original code
            sentences: Current sentences
            low_confidence_indices: Indices of sentences to refine
            
        Returns:
            Refined summary
        """
        if not low_confidence_indices:
            return ' '.join(sentences)
        
        # Create a focused prompt for refinement
        current_summary = ' '.join(sentences)
        low_conf_sentences = [sentences[i] for i in low_confidence_indices]
        
        refinement_prompt = f"""Improve this docstring by making it more accurate and clear.

Code:
```python
{code}
```

Current docstring:
{current_summary}

Focus on improving these parts: {', '.join(low_conf_sentences)}

Write improved docstring (2-3 sentences, plain English):
"""
        
        # Generate refined version
        inputs = self.tokenizer(
            refinement_prompt,
            return_tensors='pt',
            truncation=True,
            max_length=512
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                min_new_tokens=20,
                temperature=0.5,  # Lower temperature for more focused refinement
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        refined_summary = self._extract_summary(full_text, refinement_prompt)
        
        return refined_summary
    
    def generate_with_uncertainty(self, code: str, initial_summary: str) -> Dict:
        """
        Main pipeline: generate summary with uncertainty quantification.
        
        Args:
            code: Source code
            initial_summary: Initial summary from base model
            
        Returns:
            Dictionary with final summary and uncertainty metadata
        """
        # Create prompt for MC sampling
        prompt = f"""Generate a concise docstring for this function.

Function code:
```python
{code}
```

Docstring (1-2 sentences):
"""
        
        print(f"[UncertaintyAgent] Generating {self.n_samples} MC samples...")
        
        # Step 1: Generate multiple summaries with MC Dropout
        summaries = self.generate_multiple_summaries(prompt, self.n_samples)
        
        # Step 2: Calculate confidence scores
        confidence_data = self.compute_confidence_scores(summaries)
        
        print(f"[UncertaintyAgent] Mean confidence: {confidence_data['mean_confidence']:.3f}")
        print(f"[UncertaintyAgent] Low-confidence sentences: {len(confidence_data['low_confidence_indices'])}")
        
        # Step 3: Decide if refinement is needed
        refinement_applied = False
        final_summary = ' '.join(confidence_data['sentences'])
        
        if confidence_data['low_confidence_indices'] and confidence_data['mean_confidence'] < 0.8:
            print(f"[UncertaintyAgent] Refining low-confidence parts...")
            final_summary = self.refine_uncertain_parts(
                code, 
                confidence_data['sentences'],
                confidence_data['low_confidence_indices']
            )
            refinement_applied = True
        
        return {
            'final_summary': final_summary,
            'confidence_scores': confidence_data['confidence_scores'],
            'mean_confidence': confidence_data['mean_confidence'],
            'min_confidence': confidence_data['min_confidence'],
            'uncertainty_metadata': {
                'n_samples': self.n_samples,
                'low_confidence_indices': confidence_data['low_confidence_indices'],
                'refinement_applied': refinement_applied,
                'all_summaries': summaries  # For debugging
            }
        }
