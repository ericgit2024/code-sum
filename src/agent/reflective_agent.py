"""
Reflective Agent for analyzing and improving draft summaries.
"""

import torch
from typing import Dict, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer


class ReflectiveAgent:
    """Agent that critiques and refines code summaries."""
    
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, config: Dict):
        """
        Initialize reflective agent.
        
        Args:
            model: Language model
            tokenizer: Tokenizer
            config: Configuration dictionary
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.max_iterations = config['reflective_agent']['max_iterations']
        self.threshold_score = config['reflective_agent']['threshold_score']
        self.temperature = config['reflective_agent']['temperature']
        self.criteria = config['reflective_agent']['criteria']
        
    def critique_summary(self, code: str, draft_summary: str) -> str:
        """
        Generate critique of draft summary.
        
        Args:
            code: Original code
            draft_summary: Draft summary to critique
            
        Returns:
            Critique feedback
        """
        # Format critique prompt
        critique_prompt = self.config['prompts']['reflective_agent_prompt'].format(
            code=code,
            draft_summary=draft_summary
        )
        
        # Generate critique
        feedback = self._generate(critique_prompt)
        
        return feedback
    
    def refine_summary(self, code: str, draft_summary: str, feedback: str) -> str:
        """
        Refine summary based on feedback.
        
        Args:
            code: Original code
            draft_summary: Draft summary
            feedback: Critique feedback
            
        Returns:
            Refined summary
        """
        # Format refinement prompt
        refinement_prompt = self.config['prompts']['refinement_prompt'].format(
            code=code,
            draft_summary=draft_summary,
            feedback=feedback
        )
        
        # Generate refined summary
        refined = self._generate(refinement_prompt)
        
        return refined
    
    def is_approved(self, feedback: str) -> bool:
        """
        Check if summary is approved.
        
        Args:
            feedback: Critique feedback
            
        Returns:
            True if approved, False otherwise
        """
        return "APPROVED" in feedback.upper()
    
    def iterative_refinement(self, code: str, initial_summary: str) -> Tuple[str, int]:
        """
        Iteratively refine summary until approved or max iterations.
        
        Args:
            code: Original code
            initial_summary: Initial draft summary
            
        Returns:
            Tuple of (final_summary, num_iterations)
        """
        current_summary = initial_summary
        
        for iteration in range(self.max_iterations):
            # Get critique
            feedback = self.critique_summary(code, current_summary)
            
            # Check if approved
            if self.is_approved(feedback):
                print(f"Summary approved after {iteration + 1} iteration(s)")
                return current_summary, iteration + 1
            
            # Refine summary
            current_summary = self.refine_summary(code, current_summary, feedback)
            
            print(f"Iteration {iteration + 1}: Refined summary")
        
        print(f"Max iterations ({self.max_iterations}) reached")
        return current_summary, self.max_iterations
    
    def _generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return generated_text.strip()
