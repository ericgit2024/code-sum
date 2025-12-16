"""
Entity-based refinement agent for improving code summaries.
Replaces the reflective agent with a simpler entity verification approach.
"""

import torch
from typing import Dict, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

# Entity verification imports
try:
    from src.verification.entity_verifier import EntityVerifier
    from src.verification.instruction_agent import InstructionAgent
    ENTITY_VERIFICATION_AVAILABLE = True
except ImportError:
    ENTITY_VERIFICATION_AVAILABLE = False


class EntityRefinementAgent:
    """Agent that verifies entities and refines summaries based on entity feedback."""
    
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, config: Dict, eval_mode: bool = False):
        """
        Initialize entity refinement agent.
        
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
        if eval_mode and 'max_iterations_eval' in config.get('entity_verification', {}):
            self.max_iterations = config['entity_verification']['max_iterations_eval']
        elif 'max_iterations' in config.get('entity_verification', {}):
            self.max_iterations = config['entity_verification']['max_iterations']
        else:
            # Fallback to reflective_agent config for backward compatibility
            if eval_mode and 'max_iterations_eval' in config.get('reflective_agent', {}):
                self.max_iterations = config['reflective_agent']['max_iterations_eval']
            else:
                self.max_iterations = config.get('reflective_agent', {}).get('max_iterations', 3)
        
        # Generation settings
        self.temperature = config.get('entity_verification', {}).get('temperature', 0.7)
        self.max_tokens_refinement = config.get('entity_verification', {}).get('max_tokens_refinement', 300)
        
        # Fast mode settings
        self.fast_mode = config.get('entity_verification', {}).get('fast_mode', False)
        self.greedy_decoding = config.get('entity_verification', {}).get('greedy_decoding', False)
        
        # Initialize entity verification components
        if not ENTITY_VERIFICATION_AVAILABLE:
            raise ImportError("Entity verification components not available. Please check imports.")
        
        self.entity_verifier = EntityVerifier(config)
        self.instruction_agent = InstructionAgent(config)
        
    def refine_summary(self, code: str, draft_summary: str, feedback: str) -> str:
        """
        Refine summary based on entity verification feedback.
        
        Args:
            code: Original code
            draft_summary: Draft summary
            feedback: Entity verification feedback
            
        Returns:
            Refined summary
        """
        # Format refinement prompt using instruction agent's prompt
        refinement_prompt = f"""Improve this docstring based on the feedback below.

Code:
{code}

Current docstring:
{draft_summary}

Feedback:
{feedback}

Write an improved docstring that addresses all the feedback. Be accurate and complete.

Improved docstring:"""
        
        # Generate refined summary
        refined = self._generate(refinement_prompt, max_new_tokens=self.max_tokens_refinement)
        
        # Clean the output
        refined = self._clean_summary(refined)
        
        return refined
    
    def iterative_refinement(self, code: str, initial_summary: str) -> Tuple[str, int, Dict]:
        """
        Iteratively refine summary based on entity verification.
        
        Args:
            code: Original code
            initial_summary: Initial draft summary
            
        Returns:
            Tuple of (final_summary, num_iterations, metadata)
        """
        current_summary = initial_summary
        best_summary = initial_summary
        previous_summaries = []
        
        # Track verification results across iterations
        verification_history = []
        
        for iteration in range(self.max_iterations):
            print(f"\n[ITERATION {iteration + 1}] Verifying entities...")
            
            # Run entity verification
            entity_result = self.entity_verifier.verify(code, current_summary)
            
            # Track verification result
            verification_history.append({
                'iteration': iteration + 1,
                'hallucination_score': entity_result.hallucination_score,
                'precision': entity_result.precision,
                'recall': entity_result.recall,
                'f1_score': entity_result.f1_score,
                'passes': entity_result.passes_threshold
            })
            
            # Log verification status
            print(f"[ENTITY VERIFICATION] Hallucination Score: {entity_result.hallucination_score:.3f} "
                  f"(threshold: {entity_result.threshold:.2f})")
            print(f"[ENTITY VERIFICATION] Precision: {entity_result.precision:.3f}, "
                  f"Recall: {entity_result.recall:.3f}, F1: {entity_result.f1_score:.3f}")
            
            # Check if verification passed
            if entity_result.passes_threshold:
                print(f"✓ Entity verification PASSED after {iteration + 1} iteration(s)")
                metadata = {
                    'verification_history': verification_history,
                    'final_hallucination_score': entity_result.hallucination_score,
                    'final_precision': entity_result.precision,
                    'final_recall': entity_result.recall,
                    'final_f1': entity_result.f1_score,
                    'stop_reason': 'entity_verification_passed',
                    'entity_verification': self.instruction_agent.get_feedback_summary(entity_result)
                }
                return current_summary, iteration + 1, metadata
            
            # Verification failed - generate feedback
            print(f"✗ Entity verification FAILED")
            if entity_result.false_positives:
                print(f"  Hallucinated: {', '.join(sorted(list(entity_result.false_positives)[:5]))}")
            if entity_result.false_negatives:
                print(f"  Missing: {', '.join(sorted(list(entity_result.false_negatives)[:5]))}")
            
            # Generate instruction-based feedback
            feedback = self.instruction_agent.generate_instructions(entity_result, code, current_summary)
            
            print(f"\n[FEEDBACK GENERATED]")
            print(feedback[:200] + "..." if len(feedback) > 200 else feedback)
            
            # Refine summary based on feedback
            refined_summary = self.refine_summary(code, current_summary, feedback)
            
            # Check for convergence (summary not changing)
            if refined_summary in previous_summaries:
                print(f"\n⚠ Summary converged after {iteration + 1} iteration(s) (no change)")
                metadata = {
                    'verification_history': verification_history,
                    'final_hallucination_score': entity_result.hallucination_score,
                    'final_precision': entity_result.precision,
                    'final_recall': entity_result.recall,
                    'final_f1': entity_result.f1_score,
                    'stop_reason': 'converged',
                    'entity_verification': self.instruction_agent.get_feedback_summary(entity_result)
                }
                return best_summary, iteration + 1, metadata
            
            previous_summaries.append(current_summary)
            
            # Validate refined summary
            if self._is_valid_summary(refined_summary):
                current_summary = refined_summary
                # Update best summary (prefer higher F1 score)
                if self._is_better_summary(refined_summary, best_summary, entity_result):
                    best_summary = refined_summary
            else:
                print(f"⚠ Warning: Iteration {iteration + 1} produced invalid summary, keeping previous")
            
            print(f"→ Refined summary: {current_summary[:100]}...")
        
        # Max iterations reached
        print(f"\n⚠ Max iterations ({self.max_iterations}) reached")
        
        # Final verification
        final_entity_result = self.entity_verifier.verify(code, best_summary)
        
        metadata = {
            'verification_history': verification_history,
            'final_hallucination_score': final_entity_result.hallucination_score,
            'final_precision': final_entity_result.precision,
            'final_recall': final_entity_result.recall,
            'final_f1': final_entity_result.f1_score,
            'stop_reason': 'max_iterations',
            'entity_verification': self.instruction_agent.get_feedback_summary(final_entity_result)
        }
        return best_summary, self.max_iterations, metadata
    
    def _generate(self, prompt: str, max_new_tokens: int = 256, min_new_tokens: int = 0) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            min_new_tokens: Minimum tokens to generate
            
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
                'min_new_tokens': min_new_tokens,
                'do_sample': False,
                'pad_token_id': self.tokenizer.pad_token_id,
                'eos_token_id': self.tokenizer.eos_token_id
            }
        else:
            # Sampling for quality
            gen_kwargs = {
                'max_new_tokens': max_new_tokens,
                'min_new_tokens': min_new_tokens,
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
        prompt_markers = ['Feedback:', 'Code:', 'Summary:', 'Docstring:', 'Output:', 
                         'Improved docstring:', 'Improved Docstring:', 'Revised Docstring:', 
                         'Explanation:', 'Write an improved', 'Write a revised']
        for marker in prompt_markers:
            if marker in text:
                # Take only the part after the marker if it's at the start, otherwise before
                parts = text.split(marker)
                if text.strip().startswith(marker):
                    text = parts[1].strip() if len(parts) > 1 else text
                else:
                    text = parts[0].strip()
        
        # Remove common code artifacts
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip empty lines
            if not line:
                continue
            # Remove bullet points and dashes at the start
            if line.startswith(('- ', '* ', '• ')):
                line = line[2:].strip()
            # Only skip lines that START with code keywords
            if line.startswith(('def ', 'class ', 'import ', 'from ', '>>>', '```')):
                continue
            # Skip lines that are ONLY triple quotes
            if line in ['"""', "'''"]:
                continue
            # Skip lines that are just config keys or fragments
            if line.endswith("']") or line.endswith('['):
                continue
            cleaned_lines.append(line)
        
        # Convert to sentences
        sentences = []
        for line in cleaned_lines:
            if line and not line[-1] in '.!?':
                sentences.append(line + '.')
            else:
                sentences.append(line)
        
        # Join sentences with space
        cleaned = ' '.join(sentences)
        
        # Remove multiple spaces
        while '  ' in cleaned:
            cleaned = cleaned.replace('  ', ' ')
        
        # Fix double periods
        cleaned = cleaned.replace('..', '.')
        
        # Deduplicate repetitive sentences
        cleaned = self._deduplicate_sentences(cleaned)
        
        # Enforce maximum length (3-4 sentences for conciseness)
        cleaned = self._enforce_max_sentences(cleaned, max_sentences=4)
        
        return cleaned.strip()
    
    def _deduplicate_sentences(self, text: str) -> str:
        """
        Remove repetitive sentences from text with fuzzy matching.
        
        Args:
            text: Text that may contain repetition
            
        Returns:
            Deduplicated text
        """
        # Split into sentences
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        if not sentences:
            return text
        
        # Keep only unique sentences in order with fuzzy matching
        seen = set()
        unique_sentences = []
        
        for sentence in sentences:
            # Normalize for comparison (remove extra spaces, lowercase)
            normalized = ' '.join(sentence.lower().split())
            
            # Check if this is substantially similar to any seen sentence
            is_duplicate = False
            for seen_sent in seen:
                # If 80% of words overlap, consider it a duplicate
                words1 = set(normalized.split())
                words2 = set(seen_sent.split())
                if len(words1) > 0 and len(words2) > 0:
                    overlap = len(words1 & words2) / max(len(words1), len(words2))
                    if overlap > 0.8:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                seen.add(normalized)
                unique_sentences.append(sentence)
        
        # Rejoin with periods
        result = '. '.join(unique_sentences)
        if result and not result.endswith('.'):
            result += '.'
        
        return result
    
    def _enforce_max_sentences(self, text: str, max_sentences: int = 4) -> str:
        """
        Limit summary to maximum number of sentences for conciseness.
        
        Args:
            text: Input text
            max_sentences: Maximum number of sentences to keep
            
        Returns:
            Truncated text
        """
        # Split into sentences
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        # Keep only first N sentences
        if len(sentences) > max_sentences:
            sentences = sentences[:max_sentences]
        
        # Rejoin
        result = '. '.join(sentences)
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
    
    def _is_better_summary(self, new_summary: str, current_best: str, entity_result) -> bool:
        """
        Determine if new summary is better than current best based on entity verification.
        
        Args:
            new_summary: New summary to evaluate
            current_best: Current best summary
            entity_result: EntityVerificationResult for new summary
            
        Returns:
            True if new is better, False otherwise
        """
        # Verify current best
        best_result = self.entity_verifier.verify("", current_best)  # Code not needed for comparison
        
        # Prefer summary with higher F1 score
        if entity_result.f1_score > best_result.f1_score:
            return True
        
        # If F1 scores are similar, prefer lower hallucination score
        if abs(entity_result.f1_score - best_result.f1_score) < 0.05:
            return entity_result.hallucination_score < best_result.hallucination_score
        
        return False
