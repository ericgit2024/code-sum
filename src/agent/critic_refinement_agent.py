"""
Critic-based Refinement Agent that uses the Summary Critic Agent
to verify and refine code summaries based on parameter coverage.
"""

import torch
from typing import Dict, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import the critic agent
try:
    from src.verification.summary_critic_agent import SummaryCriticAgent
    CRITIC_AVAILABLE = True
except ImportError:
    CRITIC_AVAILABLE = False


class CriticRefinementAgent:
    """
    Agent that uses the Summary Critic to verify parameter coverage
    and refine summaries when needed.
    """
    
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, 
                 config: Dict, eval_mode: bool = False):
        """
        Initialize critic refinement agent.
        
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
        
        # Get critic agent config
        critic_config = config.get('summary_critic', {})
        self.max_iterations = critic_config.get('max_iterations', 3)
        self.temperature = critic_config.get('temperature', 0.7)
        
        # Fast mode settings
        self.fast_mode = critic_config.get('fast_mode', False)
        self.greedy_decoding = critic_config.get('greedy_decoding', False)
        self.max_tokens_refinement = critic_config.get('max_tokens_refinement', 300)
        
        # Initialize critic agent
        if CRITIC_AVAILABLE:
            self.critic_agent = SummaryCriticAgent(critic_config)
        else:
            self.critic_agent = None
            print("Warning: Summary Critic Agent not available")
    
    def iterative_refinement(self, code: str, initial_summary: str) -> Tuple[str, int, Dict]:
        """
        Iteratively refine summary based on critic agent feedback.
        
        Args:
            code: Original code
            initial_summary: Initial draft summary
            
        Returns:
            Tuple of (final_summary, num_iterations, metadata)
        """
        if not self.critic_agent:
            # No critic available, return initial summary
            return initial_summary, 0, {'stop_reason': 'no_critic'}
        
        current_summary = initial_summary
        best_summary = initial_summary
        
        # Track analysis across iterations
        analyses_history = []
        
        for iteration in range(self.max_iterations):
            # Analyze current summary with critic
            analysis = self.critic_agent.analyze(code, current_summary)
            analyses_history.append({
                'iteration': iteration + 1,
                'needs_regeneration': analysis.needs_regeneration,
                'confidence_score': analysis.confidence_score,
                'explained_params': list(analysis.explained_parameters),
                'unexplained_params': list(analysis.unexplained_parameters),
                'instruction': analysis.instruction
            })
            
            # Debug output
            print(f"\n[CRITIC] Iteration {iteration + 1}:")
            print(f"  Needs Regeneration: {analysis.needs_regeneration}")
            print(f"  Confidence: {analysis.confidence_score:.2f}")
            print(f"  Explained: {analysis.explained_parameters}")
            print(f"  Unexplained: {analysis.unexplained_parameters}")
            
            # Check if summary is adequate
            if not analysis.needs_regeneration:
                print(f"Summary approved after {iteration + 1} iteration(s)")
                metadata = {
                    'analyses_history': analyses_history,
                    'final_confidence': analysis.confidence_score,
                    'stop_reason': 'approved'
                }
                return current_summary, iteration + 1, metadata
            
            # Generate refinement using critic's instruction
            print(f"  Instruction: {analysis.instruction}")
            refined_summary = self.refine_summary(code, current_summary, analysis.instruction)
            
            # Validate refined summary
            if self._is_valid_summary(refined_summary):
                # Update summaries
                current_summary = refined_summary
                # Keep best based on confidence
                if analysis.confidence_score > 0.5:  # Reasonable threshold
                    best_summary = refined_summary
            else:
                print(f"Warning: Iteration {iteration + 1} produced invalid summary, keeping previous")
            
            print(f"Iteration {iteration + 1}: Refined summary")
        
        print(f"Max iterations ({self.max_iterations}) reached")
        
        # Final analysis
        final_analysis = self.critic_agent.analyze(code, best_summary)
        
        metadata = {
            'analyses_history': analyses_history,
            'final_confidence': final_analysis.confidence_score,
            'stop_reason': 'max_iterations'
        }
        return best_summary, self.max_iterations, metadata
    
    def refine_summary(self, code: str, draft_summary: str, instruction: str) -> str:
        """
        Refine summary based on critic's instruction.
        
        Args:
            code: Original code
            draft_summary: Current draft summary
            instruction: Refinement instruction from critic
            
        Returns:
            Refined summary
        """
        # Format refinement prompt - explicitly forbid docstring syntax
        refinement_prompt = f"""Rewrite this description in plain English. Do NOT use docstring syntax or code formatting.

Code:
{code}

Current description:
{draft_summary}

Feedback:
{instruction}

Write an improved docstring that addresses the feedback. Write in natural language, 2-4 sentences:
"""
        
        # Generate refined summary
        refined = self._generate(refinement_prompt, max_new_tokens=self.max_tokens_refinement)
        
        # Clean the output
        refined = self._clean_summary(refined)
        
        return refined
    
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
        
        # Determine generation parameters
        if self.fast_mode or self.greedy_decoding:
            gen_kwargs = {
                'max_new_tokens': max_new_tokens,
                'do_sample': False,
                'pad_token_id': self.tokenizer.pad_token_id,
                'eos_token_id': self.tokenizer.eos_token_id
            }
        else:
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
        Clean generated summary by aggressively removing docstring syntax.
        
        Args:
            text: Raw generated text
            
        Returns:
            Cleaned summary
        """
        import re
        
        # Remove triple quotes at start/end
        text = text.strip()
        if text.startswith('"""') or text.startswith("'''"):
            text = text[3:]
        if text.endswith('"""') or text.endswith("'''"):
            text = text[:-3]
        text = text.strip()
        
        # Remove docstring parameter syntax lines
        # Pattern: :param name: description or :type name: type
        text = re.sub(r':param\s+\w+:.*?(?=:param|:type|:return|:rtype|$)', '', text, flags=re.DOTALL)
        text = re.sub(r':type\s+\w+:.*?(?=:param|:type|:return|:rtype|$)', '', text, flags=re.DOTALL)
        text = re.sub(r':return:.*?(?=:param|:type|:rtype|$)', '', text, flags=re.DOTALL)
        text = re.sub(r':rtype:.*?(?=:param|:type|:return|$)', '', text, flags=re.DOTALL)
        
        # Remove "Examples:" section and everything after
        if 'Examples:' in text or 'Example:' in text:
            text = re.split(r'Examples?:', text)[0]
        
        # Remove prompt markers
        prompt_markers = ['Feedback:', 'Code:', 'Summary:', 'Docstring:', 'Output:', 
                         'Improved docstring:', 'Explanation:', 'Write', 'Description:',
                         'Current description:', 'Rewrite']
        for marker in prompt_markers:
            if marker in text:
                parts = text.split(marker)
                if text.strip().startswith(marker):
                    text = parts[1].strip() if len(parts) > 1 else text
                else:
                    text = parts[0].strip()
        
        # Process line by line
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Skip lines that are docstring syntax
            if line.startswith((':param', ':type', ':return', ':rtype', ':raises', ':note')):
                continue
            
            # Skip lines with only triple quotes
            if line in ['"""', "'''", '"""."""', "'''.''"]:
                continue
            
            # Remove bullet points
            if line.startswith(('- ', '* ', '• ')):
                line = line[2:].strip()
            
            # Skip code lines
            if line.startswith(('def ', 'class ', 'import ', 'from ', '>>>', '```')):
                continue
            
            # Skip example code (lines with brackets, equals, etc.)
            if re.match(r'^\[.*\]$', line):  # [1, 2, 3]
                continue
            
            cleaned_lines.append(line)
        
        # Join into paragraph
        result = ' '.join(cleaned_lines)
        
        # Remove any remaining docstring artifacts
        result = result.replace('"""', '').replace("'''", '')
        
        # Remove multiple spaces
        while '  ' in result:
            result = result.replace('  ', ' ')
        
        # Remove multiple periods
        while '..' in result:
            result = result.replace('..', '.')
        
        # Limit to 4 sentences max
        sentences = [s.strip() for s in result.split('.') if s.strip()]
        if len(sentences) > 4:
            sentences = sentences[:4]
        
        result = '. '.join(sentences)
        if result and not result.endswith('.'):
            result += '.'
        
        return result.strip()
    
    def _is_valid_summary(self, summary: str) -> bool:
        """
        Check if summary is valid (natural language, no code/docstring syntax).
        
        Args:
            summary: Summary to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not summary or len(summary.strip()) == 0:
            return False
        
        words = summary.split()
        if len(words) < 3:
            return False
        
        # Check for docstring syntax
        docstring_indicators = [':param', ':type', ':return', ':rtype', ':raises', 
                               '"""', "'''", ':note', ':example']
        if any(indicator in summary for indicator in docstring_indicators):
            return False
        
        # Check it's not code
        code_indicators = ['def ', 'class ', 'import ', 'from ', '```', 'self.', 'config[']
        if any(indicator in summary for indicator in code_indicators):
            return False
        
        return True
