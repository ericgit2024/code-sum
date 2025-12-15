"""
Iteration Agent for single-pass docstring validation and refinement.

This agent validates generated docstrings against code and structural signals,
produces targeted edit instructions, and performs constrained refinement in a
single pass while preserving all words from the initial summary.
"""

import torch
import re
import ast
from typing import Dict, Tuple, List, Set
from transformers import AutoModelForCausalLM, AutoTokenizer


class IterationAgent:
    """Agent that validates and refines code summaries in a single pass."""
    
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, config: Dict):
        """
        Initialize iteration agent.
        
        Args:
            model: Language model
            tokenizer: Tokenizer
            config: Configuration dictionary
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        # Validation settings
        self.validation_config = config['iteration_agent']['validation']
        
        # Instruction generation settings
        self.instruction_config = config['iteration_agent']['instruction_generation']
        
        # Constrained refinement settings
        self.refinement_config = config['iteration_agent']['constrained_refinement']
        
    def validate_docstring(self, code: str, docstring: str, structure_summary: str) -> Dict[str, List[str]]:
        """
        Validate docstring against code and structural signals.
        
        Args:
            code: Original code
            docstring: Generated docstring to validate
            structure_summary: Compact structure summary
            
        Returns:
            Dictionary of validation issues by category
        """
        issues = {
            'missing_parameters': [],
            'missing_return': [],
            'missing_control_flow': [],
            'missing_function_calls': [],
            'naturalness_issues': []
        }
        
        # Parse code to extract signals
        code_signals = self._extract_code_signals(code)
        structure_signals = self._parse_structure_summary(structure_summary)
        
        # Check parameters
        if self.validation_config['check_parameters']:
            for param in code_signals['parameters']:
                if param not in docstring.lower() and param != 'self':
                    issues['missing_parameters'].append(param)
        
        # Check return value
        if self.validation_config['check_return_value']:
            if code_signals['has_return'] and 'return' not in docstring.lower():
                issues['missing_return'].append('Return value not mentioned')
        
        # Check control flow
        if self.validation_config['check_control_flow']:
            if structure_signals['conditionals'] > 0 and not any(word in docstring.lower() for word in ['if', 'when', 'condition', 'check', 'validate']):
                issues['missing_control_flow'].append(f"Has {structure_signals['conditionals']} conditionals but not mentioned")
            
            if structure_signals['loops'] > 0 and not any(word in docstring.lower() for word in ['loop', 'iterate', 'each', 'all', 'every']):
                issues['missing_control_flow'].append(f"Has {structure_signals['loops']} loops but not mentioned")
            
            if structure_signals['exceptions'] > 0 and not any(word in docstring.lower() for word in ['error', 'exception', 'handle', 'catch']):
                issues['missing_control_flow'].append(f"Has exception handling but not mentioned")
        
        # Check important function calls
        if self.validation_config['check_function_calls']:
            important_calls = structure_signals['function_calls'][:3]  # Top 3
            mentioned_calls = sum(1 for call in important_calls if call.lower() in docstring.lower())
            if important_calls and mentioned_calls == 0:
                issues['missing_function_calls'].append(f"Calls {', '.join(important_calls)} but not mentioned")
        
        # Check naturalness (no code syntax)
        if self.validation_config['check_naturalness']:
            code_patterns = [r'\bdef\b', r'\bclass\b', r'\bimport\b', r'\bself\.\w+', r'==', r'!=', r'\[\w+\]']
            for pattern in code_patterns:
                if re.search(pattern, docstring):
                    issues['naturalness_issues'].append(f"Contains code syntax: {pattern}")
        
        return issues
    
    def generate_edit_instructions(self, docstring: str, validation_issues: Dict[str, List[str]]) -> str:
        """
        Generate targeted edit instructions based on validation issues.
        
        Args:
            docstring: Current docstring
            validation_issues: Issues found during validation
            
        Returns:
            Edit instructions (3-4 sentences)
        """
        # Count total issues
        total_issues = sum(len(issues) for issues in validation_issues.values())
        
        # If no issues, return empty instructions
        if total_issues == 0:
            return ""
        
        # Format validation report for prompt
        validation_report = self._format_validation_report(validation_issues)
        
        # Generate instructions using LLM
        prompt = self.config['prompts']['edit_instruction_prompt'].format(
            docstring=docstring,
            validation_report=validation_report
        )
        
        instructions = self._generate(
            prompt,
            max_new_tokens=self.instruction_config['max_instructions'] * 30,  # ~30 tokens per sentence
            temperature=self.instruction_config['temperature']
        )
        
        # Clean and limit to 3-4 sentences
        instructions = self._clean_instructions(instructions)
        
        return instructions
    
    def constrained_refinement(self, original_docstring: str, edit_instructions: str, code: str) -> str:
        """
        Refine docstring with constraint to preserve all original words.
        
        Args:
            original_docstring: Initial docstring
            edit_instructions: What to add/clarify
            code: Original code for context
            
        Returns:
            Refined docstring with all original words preserved
        """
        # If no instructions, return original
        if not edit_instructions or edit_instructions.strip() == "":
            return original_docstring
        
        # Generate refined version
        prompt = self.config['prompts']['constrained_refinement_prompt'].format(
            original_docstring=original_docstring,
            edit_instructions=edit_instructions,
            code=code
        )
        
        refined = self._generate(
            prompt,
            max_new_tokens=self.refinement_config['max_new_tokens'],
            temperature=self.refinement_config['temperature']
        )
        
        # Clean the output
        refined = self._clean_summary(refined)
        
        # Verify word preservation
        if self.refinement_config['strategy'] == 'additive':
            if not self._verify_word_preservation(original_docstring, refined):
                print("[WARNING] Word preservation violated, returning original docstring")
                return original_docstring
        
        return refined
    
    def iterate_once(self, code: str, initial_summary: str, structure_summary: str = "") -> Tuple[str, Dict]:
        """
        Perform single-pass validation and refinement.
        
        Args:
            code: Original code
            initial_summary: Initial generated docstring
            structure_summary: Compact structure summary
            
        Returns:
            Tuple of (final_summary, metadata)
        """
        print("\n[ITERATION AGENT] Starting single-pass validation and refinement...")
        
        # Step 1: Validate
        validation_issues = self.validate_docstring(code, initial_summary, structure_summary)
        total_issues = sum(len(issues) for issues in validation_issues.values())
        
        print(f"[VALIDATION] Found {total_issues} issues")
        for category, issues in validation_issues.items():
            if issues:
                print(f"  - {category}: {issues}")
        
        # Step 2: Generate edit instructions
        if total_issues > 0:
            edit_instructions = self.generate_edit_instructions(initial_summary, validation_issues)
            print(f"[INSTRUCTIONS] Generated: {edit_instructions[:100]}...")
            
            # Step 3: Constrained refinement
            refined_summary = self.constrained_refinement(initial_summary, edit_instructions, code)
            
            # Verify word preservation
            preserved = self._verify_word_preservation(initial_summary, refined_summary)
            print(f"[REFINEMENT] Word preservation: {preserved}")
            
            metadata = {
                'validation_issues': validation_issues,
                'total_issues': total_issues,
                'edit_instructions': edit_instructions,
                'word_preservation': preserved,
                'refined': True
            }
            
            return refined_summary, metadata
        else:
            print("[VALIDATION] No issues found, returning initial summary")
            metadata = {
                'validation_issues': validation_issues,
                'total_issues': 0,
                'edit_instructions': "",
                'word_preservation': True,
                'refined': False
            }
            return initial_summary, metadata
    
    def _extract_code_signals(self, code: str) -> Dict:
        """Extract signals from code using AST parsing."""
        signals = {
            'function_name': '',
            'parameters': [],
            'has_return': False,
            'return_type': None
        }
        
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    signals['function_name'] = node.name
                    signals['parameters'] = [arg.arg for arg in node.args.args]
                    if node.returns:
                        signals['return_type'] = ast.unparse(node.returns)
                    
                    # Check for return statements
                    for child in ast.walk(node):
                        if isinstance(child, ast.Return) and child.value is not None:
                            signals['has_return'] = True
                            break
                    break
        except:
            pass
        
        return signals
    
    def _parse_structure_summary(self, structure_summary: str) -> Dict:
        """Parse structure summary to extract signals."""
        signals = {
            'conditionals': 0,
            'loops': 0,
            'exceptions': 0,
            'function_calls': []
        }
        
        # Extract counts using regex
        conditional_match = re.search(r'(\d+)\s+conditional', structure_summary)
        if conditional_match:
            signals['conditionals'] = int(conditional_match.group(1))
        
        loop_match = re.search(r'(\d+)\s+loop', structure_summary)
        if loop_match:
            signals['loops'] = int(loop_match.group(1))
        
        exception_match = re.search(r'(\d+)\s+exception', structure_summary)
        if exception_match:
            signals['exceptions'] = int(exception_match.group(1))
        
        # Extract function calls
        calls_match = re.search(r'calls\s+\[(.*?)\]', structure_summary)
        if calls_match:
            calls_str = calls_match.group(1)
            signals['function_calls'] = [call.strip() for call in calls_str.split(',')]
        
        return signals
    
    def _format_validation_report(self, issues: Dict[str, List[str]]) -> str:
        """Format validation issues into a report string."""
        report_lines = []
        
        if issues['missing_parameters']:
            report_lines.append(f"Missing parameters: {', '.join(issues['missing_parameters'])}")
        
        if issues['missing_return']:
            report_lines.append("Missing return value description")
        
        if issues['missing_control_flow']:
            for issue in issues['missing_control_flow']:
                report_lines.append(f"Control flow: {issue}")
        
        if issues['missing_function_calls']:
            for issue in issues['missing_function_calls']:
                report_lines.append(f"Function calls: {issue}")
        
        if issues['naturalness_issues']:
            report_lines.append("Contains code syntax (should be natural language)")
        
        return "\n".join(report_lines) if report_lines else "No issues found"
    
    def _clean_instructions(self, text: str) -> str:
        """Clean and limit edit instructions to 3-4 sentences."""
        # Remove prompt markers
        for marker in ['Instructions:', 'Edit instructions:', 'Suggestions:']:
            if marker in text:
                text = text.split(marker)[-1].strip()
        
        # Split into sentences
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        
        # Limit to 4 sentences
        sentences = sentences[:4]
        
        # Rejoin
        return '. '.join(sentences) + '.' if sentences else ""
    
    def _clean_summary(self, text: str) -> str:
        """Clean generated summary."""
        # Remove prompt markers
        prompt_markers = ['Docstring:', 'Summary:', 'Output:', 'Improved docstring:']
        for marker in prompt_markers:
            if marker in text:
                parts = text.split(marker)
                text = parts[-1].strip()
        
        # Remove code blocks
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        
        # Remove triple quotes
        text = text.replace('"""', '').replace("'''", '')
        
        # Clean up whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def _verify_word_preservation(self, original: str, refined: str) -> bool:
        """
        Verify that all words from original are present in refined.
        
        Args:
            original: Original docstring
            refined: Refined docstring
            
        Returns:
            True if all original words are preserved
        """
        # Tokenize into words (lowercase, alphanumeric only)
        original_words = set(re.findall(r'\b\w+\b', original.lower()))
        refined_words = set(re.findall(r'\b\w+\b', refined.lower()))
        
        # Check if all original words are in refined
        missing_words = original_words - refined_words
        
        if missing_words:
            print(f"[WARNING] Missing words: {missing_words}")
            return False
        
        return True
    
    def _generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.7) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
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
        gen_kwargs = {
            'max_new_tokens': max_new_tokens,
            'temperature': temperature,
            'do_sample': True,
            'top_p': 0.9,
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self.tokenizer.eos_token_id
        }
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        
        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return generated_text.strip()
