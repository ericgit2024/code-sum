"""
Reflective Agent for analyzing and improving draft summaries.
"""

import torch
from typing import Dict, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer


class ReflectiveAgent:
    """Agent that critiques and refines code summaries."""
    
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, config: Dict, eval_mode: bool = False):
        """
        Initialize reflective agent.
        
        Args:
            model: Language model
            tokenizer: Tokenizer
            config: Configuration dictionary
            eval_mode: If True, use faster settings for evaluation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.eval_mode = eval_mode
        
        # Use eval-specific iterations if in eval mode
        if eval_mode and 'max_iterations_eval' in config['reflective_agent']:
            self.max_iterations = config['reflective_agent']['max_iterations_eval']
        else:
            self.max_iterations = config['reflective_agent']['max_iterations']
        
        self.threshold_score = config['reflective_agent']['threshold_score']
        self.temperature = config['reflective_agent']['temperature']
        self.criteria = config['reflective_agent']['criteria']
        
        # Fast mode settings
        self.fast_mode = config['reflective_agent'].get('fast_mode', False)
        self.greedy_decoding = config['reflective_agent'].get('greedy_decoding', False)
        self.max_tokens_critique = config['reflective_agent'].get('max_tokens_critique', 128)
        self.max_tokens_refinement = config['reflective_agent'].get('max_tokens_refinement', 150)
        
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
        
        # Generate critique with reduced tokens
        feedback = self._generate(critique_prompt, max_new_tokens=self.max_tokens_critique)
        
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
        
        # Generate refined summary with reduced tokens
        refined = self._generate(refinement_prompt, max_new_tokens=self.max_tokens_refinement)
        
        # Clean the output
        refined = self._clean_summary(refined)
        
        return refined
    
    def is_approved(self, feedback: str) -> bool:
        """
        Check if summary is approved using multiple signals.
        
        Args:
            feedback: Critique feedback
            
        Returns:
            True if approved, False otherwise
        """
        feedback_upper = feedback.upper()
        
        # Rejection signals (check first, override approval)
        rejection_keywords = [
            "NOT APPROVED",
            "NEEDS IMPROVEMENT",
            "MISSING",
            "INCORRECT",
            "WRONG",
            "INACCURATE",
            "INCOMPLETE"
        ]
        
        # Check for rejection first
        if any(keyword in feedback_upper for keyword in rejection_keywords):
            return False
        
        # Approval signals
        approval_keywords = [
            "APPROVED",
            "LOOKS GOOD",
            "GOOD",
            "ACCEPTABLE",
            "MEETS ALL CRITERIA",
            "WELL DONE",
            "SATISFACTORY",
            "CORRECT",
            "ACCURATE"
        ]
        
        # Check for approval
        return any(keyword in feedback_upper for keyword in approval_keywords)
    
    def iterative_refinement(self, code: str, initial_summary: str) -> Tuple[str, int]:
        """
        Iteratively refine summary with convergence detection and best summary tracking.
        
        Args:
            code: Original code
            initial_summary: Initial draft summary
            
        Returns:
            Tuple of (final_summary, num_iterations)
        """
        current_summary = initial_summary
        best_summary = initial_summary
        previous_summaries = []
        
        for iteration in range(self.max_iterations):
            # Get critique FIRST
            feedback = self.critique_summary(code, current_summary)
            
            # Debug output
            print(f"\n[DEBUG] Iteration {iteration + 1} Critique:")
            print(f"  {feedback[:300]}")
            
            # Check if approved
            is_approved = self.is_approved(feedback)
            print(f"[DEBUG] Approved: {is_approved}")
            
            if is_approved:
                print(f"Summary approved after {iteration + 1} iteration(s)")
                return current_summary, iteration + 1
            
            # Refine summary
            refined_summary = self.refine_summary(code, current_summary, feedback)
            
            # Check for convergence AFTER refinement
            if refined_summary in previous_summaries:
                print(f"Summary converged after {iteration + 1} iteration(s)")
                return best_summary, iteration + 1
            
            previous_summaries.append(current_summary)
            
            # Validate refined summary
            if self._is_valid_summary(refined_summary):
                current_summary = refined_summary
                # Update best summary using quality heuristics
                if self._is_better_summary(refined_summary, best_summary):
                    best_summary = refined_summary
            else:
                print(f"Warning: Iteration {iteration + 1} produced invalid summary, keeping previous")
            
            print(f"Iteration {iteration + 1}: Refined summary")
        
        print(f"Max iterations ({self.max_iterations}) reached")
        return best_summary, self.max_iterations
    
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
        
        # Determine generation parameters based on fast mode
        if self.fast_mode or self.greedy_decoding:
            # Greedy decoding for speed
            gen_kwargs = {
                'max_new_tokens': max_new_tokens,
                'do_sample': False,
                'pad_token_id': self.tokenizer.pad_token_id,
                'eos_token_id': self.tokenizer.eos_token_id
            }
        else:
            # Sampling for quality
            gen_kwargs = {
                'max_new_tokens': max_new_tokens,
                'temperature': self.temperature,
                'do_sample': True,
                'top_p': 0.9,
                'pad_token_id': self.tokenizer.pad_token_id,
                'eos_token_id': self.tokenizer.eos_token_id
            }
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        
        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return generated_text.strip()
    
    def _clean_summary(self, text: str) -> str:
        """
        Clean generated summary by removing code artifacts and formatting issues.
        
        Args:
            text: Raw generated text
            
        Returns:
            Cleaned summary
        """
        # Remove prompt markers that might leak into output
        prompt_markers = ['Feedback:', 'Code:', 'Summary:', 'Docstring:', 'Output:', 'Improved docstring:']
        for marker in prompt_markers:
            if marker in text:
                # Take only the part before the marker
                text = text.split(marker)[0].strip()
        
        # Remove common code artifacts (but be less aggressive)
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip empty lines
            if not line:
                continue
            # Only skip lines that START with code keywords
            if line.startswith(('def ', 'class ', 'import ', 'from ')):
                continue
            # Skip lines that are ONLY triple quotes
            if line in ['"""', "'''"]:
                continue
            # Skip lines that are just config keys or fragments
            if line.endswith("']") or line.endswith('['):
                continue
            cleaned_lines.append(line)
        
        # Join and clean up
        cleaned = ' '.join(cleaned_lines)
        
        # Remove multiple spaces
        while '  ' in cleaned:
            cleaned = cleaned.replace('  ', ' ')
        
        # Deduplicate repetitive sentences
        cleaned = self._deduplicate_sentences(cleaned)
        
        return cleaned.strip()
    
    def _deduplicate_sentences(self, text: str) -> str:
        """
        Remove repetitive sentences from text.
        
        Args:
            text: Text that may contain repetition
            
        Returns:
            Deduplicated text
        """
        # Split into sentences
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        if not sentences:
            return text
        
        # Keep only unique sentences in order
        seen = set()
        unique_sentences = []
        
        for sentence in sentences:
            # Normalize for comparison
            normalized = ' '.join(sentence.lower().split())
            if normalized not in seen:
                seen.add(normalized)
                unique_sentences.append(sentence)
        
        # Rejoin with periods
        result = '. '.join(unique_sentences)
        if result and not result.endswith('.'):
            result += '.'
        
        return result
    
    def _is_valid_summary(self, summary: str) -> bool:
        """
        Check if summary is valid (not empty, not corrupted).
        
        Args:
            summary: Summary to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not summary or len(summary.strip()) == 0:
            return False
        
        # Check minimum word count
        words = summary.split()
        if len(words) < 3:
            return False
        
        # Check it's not just code fragments
        code_indicators = ['def ', 'class ', 'import ', 'from ', '```', 'self.', 'config[']
        if any(indicator in summary for indicator in code_indicators):
            return False
        
        return True
    
    def _is_better_summary(self, new_summary: str, current_best: str) -> bool:
        """
        Determine if new summary is better than current best.
        
        Args:
            new_summary: New summary to evaluate
            current_best: Current best summary
            
        Returns:
            True if new is better, False otherwise
        """
        new_words = len(new_summary.split())
        best_words = len(current_best.split())
        
        # Prefer summaries with reasonable length (5-150 words)
        new_in_range = 5 <= new_words <= 150
        best_in_range = 5 <= best_words <= 150
        
        # If only one is in range, prefer that one
        if new_in_range and not best_in_range:
            return True
        if best_in_range and not new_in_range:
            return False
        
        # Both in range or both out of range: prefer more concise (but not too short)
        # Ideal length is 20-50 words
        new_distance = abs(new_words - 35)
        best_distance = abs(best_words - 35)
        
        return new_distance < best_distance
