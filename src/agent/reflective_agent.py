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
        self.max_tokens_critique = config['reflective_agent'].get('max_tokens_critique', 250)
        self.max_tokens_refinement = config['reflective_agent'].get('max_tokens_refinement', 300)
        
        # Scoring system settings
        self.scoring_enabled = config['reflective_agent'].get('scoring', {}).get('enabled', False)
        self.score_weights = config['reflective_agent'].get('scoring', {}).get('weights', {
            'accuracy': 0.35,
            'completeness': 0.30,
            'naturalness': 0.20,
            'conciseness': 0.15
        })
        self.approval_threshold = config['reflective_agent'].get('scoring', {}).get('approval_threshold', 0.75)
        self.early_stop_threshold = config['reflective_agent'].get('scoring', {}).get('early_stop_threshold', 0.90)
        self.min_improvement = config['reflective_agent'].get('scoring', {}).get('min_improvement', 0.05)
        
        # Adaptive iteration settings
        self.adaptive_enabled = config['reflective_agent'].get('adaptive_iterations', {}).get('enabled', False)
        self.complexity_thresholds = config['reflective_agent'].get('adaptive_iterations', {}).get('complexity_thresholds', {
            'simple': 1,
            'moderate': 2,
            'complex': 3
        })
        
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
        feedback = self._generate(
            critique_prompt, 
            max_new_tokens=self.max_tokens_critique
        )
        
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
        # If feedback is empty or very short, treat as approval
        # (LLM has nothing to critique = summary is good)
        if not feedback or len(feedback.strip()) < 10:
            return True
        
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
    
    def iterative_refinement(self, code: str, initial_summary: str) -> Tuple[str, int, Dict]:
        """
        Iteratively refine summary with scoring and adaptive iterations.
        
        Args:
            code: Original code
            initial_summary: Initial draft summary
            
        Returns:
            Tuple of (final_summary, num_iterations, metadata)
        """
        current_summary = initial_summary
        best_summary = initial_summary
        previous_summaries = []
        
        # Assess code complexity for adaptive iterations
        complexity_info = {'cyclomatic_complexity': 0, 'optimal_iterations': self.max_iterations}
        if self.adaptive_enabled:
            complexity_info = self._assess_code_complexity(code)
            max_iterations = complexity_info['optimal_iterations']
            print(f"[ADAPTIVE] Code complexity: {complexity_info['cyclomatic_complexity']}, "
                  f"using {max_iterations} max iterations")
        else:
            max_iterations = self.max_iterations
        
        # Track scores across iterations
        scores_history = []
        previous_score = 0.0
        
        for iteration in range(max_iterations):
            # Get critique FIRST
            feedback = self.critique_summary(code, current_summary)
            
            # Extract scores if scoring is enabled
            if self.scoring_enabled:
                scores = self._extract_scores(feedback)
                weighted_score = self._calculate_weighted_score(scores)
                scores_history.append({
                    'iteration': iteration + 1,
                    'scores': scores,
                    'weighted_score': weighted_score
                })
                
                # Debug output with scores
                print(f"\n[SCORING] Iteration {iteration + 1}:")
                print(f"  Accuracy: {scores.get('accuracy', 0):.2f} | "
                      f"Completeness: {scores.get('completeness', 0):.2f} | "
                      f"Naturalness: {scores.get('naturalness', 0):.2f} | "
                      f"Conciseness: {scores.get('conciseness', 0):.2f}")
                print(f"  Weighted Score: {weighted_score:.2f}")
                
                # Check for early stopping
                should_stop, reason = self._should_stop_early(weighted_score, previous_score, iteration)
                if should_stop:
                    print(f"[EARLY STOP] {reason}")
                    metadata = {
                        'scores_history': scores_history,
                        'final_score': weighted_score,
                        'complexity': complexity_info,
                        'stop_reason': reason
                    }
                    return current_summary, iteration + 1, metadata
                
                # Check if approved based on score
                is_approved = weighted_score >= self.approval_threshold
                print(f"  Approved: {is_approved} (threshold: {self.approval_threshold})")
                
                previous_score = weighted_score
            else:
                # Fallback to keyword-based approval
                print(f"\n[DEBUG] Iteration {iteration + 1} Critique:")
                print(f"  {feedback[:300]}")
                is_approved = self.is_approved(feedback)
                print(f"[DEBUG] Approved: {is_approved}")
            
            if is_approved:
                print(f"Summary approved after {iteration + 1} iteration(s)")
                metadata = {
                    'scores_history': scores_history,
                    'final_score': weighted_score if self.scoring_enabled else None,
                    'complexity': complexity_info,
                    'stop_reason': 'approved'
                }
                return current_summary, iteration + 1, metadata
            
            # Refine summary
            refined_summary = self.refine_summary(code, current_summary, feedback)
            
            # Check for convergence AFTER refinement
            if refined_summary in previous_summaries:
                print(f"Summary converged after {iteration + 1} iteration(s)")
                metadata = {
                    'scores_history': scores_history,
                    'final_score': weighted_score if self.scoring_enabled else None,
                    'complexity': complexity_info,
                    'stop_reason': 'converged'
                }
                return best_summary, iteration + 1, metadata
            
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
        
        print(f"Max iterations ({max_iterations}) reached")
        metadata = {
            'scores_history': scores_history,
            'final_score': weighted_score if self.scoring_enabled else None,
            'complexity': complexity_info,
            'stop_reason': 'max_iterations'
        }
        return best_summary, max_iterations, metadata
    
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
        prompt_markers = ['Feedback:', 'Code:', 'Summary:', 'Docstring:', 'Output:', 'Improved docstring:', 'Explanation:']
        for marker in prompt_markers:
            if marker in text:
                # Take only the part after the marker if it's at the start, otherwise before
                parts = text.split(marker)
                if text.strip().startswith(marker):
                    text = parts[1].strip() if len(parts) > 1 else text
                else:
                    text = parts[0].strip()
        
        # Remove common code artifacts (but be less aggressive)
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
        
        # Convert to sentences (handle bullet points that were separated by newlines)
        # Join with periods if lines don't already end with punctuation
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
    
    def _extract_scores(self, critique: str) -> Dict[str, float]:
        """
        Extract numerical scores from critique text.
        
        Args:
            critique: Critique feedback with scores
            
        Returns:
            Dictionary of criterion scores (0.0-1.0)
        """
        import re
        
        scores = {}
        
        # Try to extract scores for each criterion
        for criterion in ['accuracy', 'completeness', 'naturalness', 'conciseness']:
            # Look for patterns like "Accuracy: 0.8" or "Accuracy: [0.8]"
            pattern = rf'{criterion}\s*:\s*\[?([0-1]?\.?\d+)\]?'
            match = re.search(pattern, critique, re.IGNORECASE)
            
            if match:
                try:
                    score = float(match.group(1))
                    # Clamp to [0, 1]
                    scores[criterion] = max(0.0, min(1.0, score))
                except ValueError:
                    scores[criterion] = 0.5  # Default if parsing fails
            else:
                # Default to 0.5 if not found
                scores[criterion] = 0.5
        
        return scores
    
    def _calculate_weighted_score(self, scores: Dict[str, float]) -> float:
        """
        Calculate weighted aggregate score.
        
        Args:
            scores: Individual criterion scores
            
        Returns:
            Weighted score (0.0-1.0)
        """
        weighted_sum = 0.0
        total_weight = 0.0
        
        for criterion, weight in self.score_weights.items():
            if criterion in scores:
                weighted_sum += scores[criterion] * weight
                total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            return 0.5  # Default if no weights
    
    def _assess_code_complexity(self, code: str) -> Dict[str, any]:
        """
        Assess code complexity to determine optimal iteration count.
        
        Args:
            code: Source code to analyze
            
        Returns:
            Dictionary with complexity metrics
        """
        import ast
        
        try:
            tree = ast.parse(code)
        except:
            # If parsing fails, assume moderate complexity
            return {'cyclomatic_complexity': 5, 'optimal_iterations': 2}
        
        # Calculate cyclomatic complexity (simplified)
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            # Each decision point adds to complexity
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        
        # Determine optimal iterations based on complexity
        if complexity <= 3:
            optimal_iterations = self.complexity_thresholds.get('simple', 1)
        elif complexity <= 8:
            optimal_iterations = self.complexity_thresholds.get('moderate', 2)
        else:
            optimal_iterations = self.complexity_thresholds.get('complex', 3)
        
        return {
            'cyclomatic_complexity': complexity,
            'optimal_iterations': optimal_iterations
        }
    
    def _should_stop_early(self, current_score: float, previous_score: float, iteration: int) -> Tuple[bool, str]:
        """
        Determine if iteration should stop early.
        
        Args:
            current_score: Current weighted score
            previous_score: Previous weighted score
            iteration: Current iteration number
            
        Returns:
            Tuple of (should_stop, reason)
        """
        # Stop if score is excellent
        if current_score >= self.early_stop_threshold:
            return True, f"Excellent score achieved ({current_score:.2f})"
        
        # Stop if improvement is minimal (after first iteration)
        if iteration > 0:
            improvement = current_score - previous_score
            if improvement < self.min_improvement:
                return True, f"Minimal improvement ({improvement:.3f})"
        
        return False, ""
