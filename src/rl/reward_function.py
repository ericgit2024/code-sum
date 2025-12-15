"""
Reward function for RL-based docstring training.

Computes multi-component rewards based on:
1. Parameter coverage
2. Return value mention
3. Control flow coverage
4. Naturalness (no code syntax)
5. Fluency (BLEU-based)
6. Hallucination penalty
"""

import ast
import re
from typing import Dict, Tuple, List, Set
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


class RewardFunction:
    """Compute execution-based rewards for docstring quality."""
    
    def __init__(self, config: Dict = None):
        """
        Initialize reward function.
        
        Args:
            config: Configuration dictionary with reward weights
        """
        # Default weights
        default_weights = {
            'parameter_coverage': 0.25,
            'return_mention': 0.25,
            'control_flow': 0.20,
            'naturalness': 0.15,
            'fluency': 0.10,
            'hallucination_penalty': 0.05
        }
        
        # Load weights from config or use defaults
        if config and 'rl' in config and 'reward_weights' in config['rl']:
            self.weights = config['rl']['reward_weights']
        else:
            self.weights = default_weights
        
        # Normalize weights to sum to 1.0
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}
        
        # BLEU smoothing function
        self.smoothing = SmoothingFunction().method1
    
    def compute_reward(self, code: str, docstring: str, reference: str = None) -> Tuple[float, Dict]:
        """
        Compute total reward for (code, docstring) pair.
        
        Args:
            code: Python function code
            docstring: Generated docstring
            reference: Optional reference docstring for fluency comparison
            
        Returns:
            Tuple of (total_reward, component_breakdown)
        """
        components = {}
        
        # 1. Parameter Coverage (25%)
        components['parameter_coverage'] = self._check_parameter_coverage(code, docstring)
        
        # 2. Return Value Mention (25%)
        components['return_mention'] = self._check_return_mention(code, docstring)
        
        # 3. Control Flow Coverage (20%)
        components['control_flow'] = self._check_control_flow(code, docstring)
        
        # 4. Naturalness (15%)
        components['naturalness'] = self._check_naturalness(docstring)
        
        # 5. Fluency (10%)
        if reference:
            components['fluency'] = self._check_fluency(docstring, reference)
        else:
            components['fluency'] = 1.0  # No penalty if no reference
        
        # 6. Hallucination Penalty (5%)
        components['hallucination_penalty'] = self._check_hallucinations(code, docstring)
        
        # Compute weighted sum
        total_reward = sum(
            self.weights[key] * components[key]
            for key in components
        )
        
        return total_reward, components
    
    def _check_parameter_coverage(self, code: str, docstring: str) -> float:
        """
        Check if all function parameters are mentioned in docstring.
        
        Args:
            code: Function code
            docstring: Generated docstring
            
        Returns:
            Score from 0.0 to 1.0
        """
        try:
            # Parse code to extract parameters
            tree = ast.parse(code)
            func_def = next(
                (node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)),
                None
            )
            
            if not func_def:
                return 1.0  # Not a function, no penalty
            
            # Extract parameter names (excluding 'self')
            params = [
                arg.arg for arg in func_def.args.args 
                if arg.arg != 'self'
            ]
            
            if not params:
                return 1.0  # No parameters to check
            
            # Check how many parameters are mentioned (fuzzy matching)
            docstring_lower = docstring.lower()
            mentioned = 0
            
            for param in params:
                # Check for exact match or common variations
                param_lower = param.lower()
                patterns = [
                    param_lower,  # exact
                    f"the {param_lower}",  # "the price"
                    f"{param_lower}s",  # plural
                    f"'{param_lower}'",  # quoted
                    f'"{param_lower}"',  # double quoted
                ]
                
                if any(pattern in docstring_lower for pattern in patterns):
                    mentioned += 1
            
            return mentioned / len(params)
            
        except Exception as e:
            # If parsing fails, return neutral score
            return 0.5
    
    def _check_return_mention(self, code: str, docstring: str) -> float:
        """
        Check if return value is mentioned when function returns something.
        
        Args:
            code: Function code
            docstring: Generated docstring
            
        Returns:
            Score from 0.0 to 1.0
        """
        try:
            # Check if code has meaningful return statement
            tree = ast.parse(code)
            has_return = False
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Return):
                    # Check if it's not just "return None" or "return"
                    if node.value is not None:
                        # Check if it's not a constant None
                        if not (isinstance(node.value, ast.Constant) and node.value.value is None):
                            has_return = True
                            break
            
            if not has_return:
                return 1.0  # No return value, no need to mention
            
            # Check if docstring mentions return
            return_keywords = [
                'return', 'returns', 'output', 'result', 'gives',
                'produces', 'yields', 'provides', 'generates'
            ]
            
            docstring_lower = docstring.lower()
            mentions_return = any(kw in docstring_lower for kw in return_keywords)
            
            return 1.0 if mentions_return else 0.0
            
        except Exception:
            return 0.5
    
    def _check_control_flow(self, code: str, docstring: str) -> float:
        """
        Check if control flow structures are described in docstring.
        
        Args:
            code: Function code
            docstring: Generated docstring
            
        Returns:
            Score from 0.0 to 1.0
        """
        try:
            # Define control flow patterns and their description keywords
            control_flow_patterns = {
                'if': ['if', 'when', 'condition', 'check', 'validate', 'whether', 'case'],
                'for': ['iterate', 'loop', 'each', 'all', 'every', 'through', 'over'],
                'while': ['while', 'until', 'loop', 'repeat', 'continue'],
                'try': ['handle', 'catch', 'error', 'exception', 'raise', 'fail']
            }
            
            # Parse code to find control structures
            tree = ast.parse(code)
            structures_in_code = set()
            
            for node in ast.walk(tree):
                if isinstance(node, ast.If):
                    structures_in_code.add('if')
                elif isinstance(node, ast.For):
                    structures_in_code.add('for')
                elif isinstance(node, ast.While):
                    structures_in_code.add('while')
                elif isinstance(node, (ast.Try, ast.ExceptHandler, ast.Raise)):
                    structures_in_code.add('try')
            
            if not structures_in_code:
                return 1.0  # No control flow to describe
            
            # Check if docstring mentions them
            docstring_lower = docstring.lower()
            mentioned = 0
            
            for structure in structures_in_code:
                keywords = control_flow_patterns[structure]
                if any(kw in docstring_lower for kw in keywords):
                    mentioned += 1
            
            return mentioned / len(structures_in_code)
            
        except Exception:
            return 0.5
    
    def _check_naturalness(self, docstring: str) -> float:
        """
        Check if docstring is written in natural language (no code syntax).
        
        Args:
            docstring: Generated docstring
            
        Returns:
            Score from 0.0 to 1.0
        """
        # Code syntax patterns to penalize
        code_patterns = [
            r'def\s+\w+',           # def function_name
            r'class\s+\w+',         # class ClassName
            r'self\.\w+',           # self.attribute
            r'==|!=|<=|>=|\+=|-=',  # comparison/assignment operators
            r'\w+\[\w+\]',          # array indexing (but allow "the [value]")
            r'import\s+\w+',        # import statements
            r'from\s+\w+',          # from imports
            r'->\s*\w+',            # type hints
            r':\s*\w+\s*=',         # parameter defaults
        ]
        
        penalties = 0
        max_penalties = len(code_patterns)
        
        for pattern in code_patterns:
            if re.search(pattern, docstring):
                penalties += 1
        
        # Return score (fewer penalties = higher score)
        return 1.0 - (penalties / max_penalties)
    
    def _check_fluency(self, docstring: str, reference: str) -> float:
        """
        Check fluency using BLEU score against reference.
        
        Args:
            docstring: Generated docstring
            reference: Reference docstring
            
        Returns:
            Score from 0.0 to 1.0
        """
        try:
            # Tokenize
            hypothesis = docstring.lower().split()
            reference_tokens = reference.lower().split()
            
            # Compute BLEU-2 (bigram) for fluency
            bleu = sentence_bleu(
                [reference_tokens],
                hypothesis,
                weights=(0.5, 0.5, 0, 0),  # BLEU-2
                smoothing_function=self.smoothing
            )
            
            return bleu
            
        except Exception:
            return 0.5
    
    def _check_hallucinations(self, code: str, docstring: str) -> float:
        """
        Penalize hallucinations (mentioning non-existent parameters/behaviors).
        
        Args:
            code: Function code
            docstring: Generated docstring
            
        Returns:
            Score from 0.0 to 1.0 (1.0 = no hallucinations)
        """
        try:
            # Extract actual parameters from code
            tree = ast.parse(code)
            func_def = next(
                (node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)),
                None
            )
            
            if not func_def:
                return 1.0
            
            actual_params = {
                arg.arg.lower() for arg in func_def.args.args 
                if arg.arg != 'self'
            }
            
            # Look for potential parameter mentions in docstring
            # Match word boundaries to avoid false positives
            docstring_lower = docstring.lower()
            
            # Common parameter-like words that might be hallucinations
            # Look for words that look like variable names (lowercase, underscores)
            potential_params = re.findall(r'\b[a-z_][a-z0-9_]*\b', docstring_lower)
            
            # Filter out common English words
            common_words = {
                'the', 'a', 'an', 'and', 'or', 'but', 'if', 'when', 'then',
                'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'have', 'has', 'had', 'do', 'does', 'did',
                'will', 'would', 'should', 'could', 'may', 'might',
                'this', 'that', 'these', 'those', 'it', 'its',
                'for', 'to', 'of', 'in', 'on', 'at', 'by', 'with',
                'from', 'as', 'into', 'through', 'during', 'before', 'after',
                'function', 'method', 'class', 'object', 'value', 'values',
                'return', 'returns', 'result', 'output', 'input',
                'calculate', 'calculates', 'compute', 'computes',
                'check', 'checks', 'validate', 'validates',
                'get', 'gets', 'set', 'sets', 'add', 'adds',
                'remove', 'removes', 'delete', 'deletes',
                'create', 'creates', 'update', 'updates',
                'number', 'numbers', 'string', 'strings', 'list', 'lists',
                'dict', 'dictionary', 'tuple', 'set',
                'true', 'false', 'none', 'null',
                'first', 'second', 'third', 'last', 'next', 'previous',
                'new', 'old', 'current', 'final', 'initial',
                'max', 'min', 'sum', 'total', 'count', 'length',
                'error', 'exception', 'message', 'code', 'data'
            }
            
            # Count hallucinated parameters
            hallucinations = 0
            for param in potential_params:
                # Skip if it's a common word
                if param in common_words:
                    continue
                # Skip if it's an actual parameter
                if param in actual_params:
                    continue
                # Skip very short words (likely not parameters)
                if len(param) <= 2:
                    continue
                
                # This might be a hallucination
                hallucinations += 1
            
            # Penalize based on number of hallucinations
            # Allow 1-2 false positives before penalizing
            if hallucinations <= 2:
                return 1.0
            else:
                # Gradual penalty
                penalty = min(1.0, (hallucinations - 2) * 0.2)
                return 1.0 - penalty
                
        except Exception:
            return 1.0  # No penalty if we can't parse
    
    def compute_batch_rewards(self, code_batch: List[str], 
                             docstring_batch: List[str],
                             reference_batch: List[str] = None) -> Tuple[List[float], List[Dict]]:
        """
        Compute rewards for a batch of samples.
        
        Args:
            code_batch: List of code samples
            docstring_batch: List of generated docstrings
            reference_batch: Optional list of reference docstrings
            
        Returns:
            Tuple of (reward_list, breakdown_list)
        """
        rewards = []
        breakdowns = []
        
        for i, (code, docstring) in enumerate(zip(code_batch, docstring_batch)):
            reference = reference_batch[i] if reference_batch else None
            reward, breakdown = self.compute_reward(code, docstring, reference)
            rewards.append(reward)
            breakdowns.append(breakdown)
        
        return rewards, breakdowns
