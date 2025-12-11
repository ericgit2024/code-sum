"""
Compact structure summarizer that extracts high-level features.
"""

import ast
from typing import Dict


class CompactStructureSummarizer:
    """Extracts compact, human-readable structural summaries."""
    
    def __init__(self):
        pass
    
    def summarize_code(self, code: str) -> str:
        """
        Generate a compact structural summary of the code.
        
        Args:
            code: Python source code
            
        Returns:
            Compact summary string
        """
        try:
            tree = ast.parse(code)
        except:
            return "Structure: unable to parse code"
        
        # Extract high-level features
        features = self._extract_features(tree)
        
        # Format as natural language
        summary_parts = []
        
        # Function info
        if features['num_functions'] > 0:
            summary_parts.append(f"{features['num_functions']} function(s)")
        
        # Parameters
        if features['num_params'] > 0:
            summary_parts.append(f"{features['num_params']} parameter(s)")
        
        # Control flow
        control_flow = []
        if features['num_if'] > 0:
            control_flow.append(f"{features['num_if']} conditional(s)")
        if features['num_loops'] > 0:
            control_flow.append(f"{features['num_loops']} loop(s)")
        if features['num_try'] > 0:
            control_flow.append(f"{features['num_try']} try-except(s)")
        
        if control_flow:
            summary_parts.append(", ".join(control_flow))
        
        # Returns
        if features['has_return']:
            summary_parts.append("returns value")
        
        # Calls
        if features['num_calls'] > 0:
            summary_parts.append(f"calls {features['num_calls']} function(s)")
        
        if not summary_parts:
            return "Structure: simple code block"
        
        return "Structure: " + ", ".join(summary_parts)
    
    def _extract_features(self, tree: ast.AST) -> Dict:
        """Extract high-level structural features."""
        features = {
            'num_functions': 0,
            'num_params': 0,
            'num_if': 0,
            'num_loops': 0,
            'num_try': 0,
            'has_return': False,
            'num_calls': 0,
            'num_classes': 0
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                features['num_functions'] += 1
                features['num_params'] += len(node.args.args)
            elif isinstance(node, ast.ClassDef):
                features['num_classes'] += 1
            elif isinstance(node, ast.If):
                features['num_if'] += 1
            elif isinstance(node, (ast.For, ast.While)):
                features['num_loops'] += 1
            elif isinstance(node, ast.Try):
                features['num_try'] += 1
            elif isinstance(node, ast.Return):
                features['has_return'] = True
            elif isinstance(node, ast.Call):
                features['num_calls'] += 1
        
        return features
