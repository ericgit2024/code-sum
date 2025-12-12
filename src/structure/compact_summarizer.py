"""
Compact structure summarizer that extracts high-level features.
"""

import ast
from typing import Dict, List


class CompactStructureSummarizer:
    """Extracts compact, human-readable structural summaries with enhanced information."""
    
    def __init__(self):
        pass
    
    def summarize_code(self, code: str) -> str:
        """
        Generate an enhanced compact structural summary of the code.
        
        Args:
            code: Python source code
            
        Returns:
            Enhanced compact summary string
        """
        try:
            tree = ast.parse(code)
        except:
            return "Structure: unable to parse code"
        
        # Extract enhanced features
        features = self._extract_enhanced_features(tree)
        
        # Format as natural language with richer information
        summary_parts = []
        
        # Function info with name
        if features['function_name']:
            func_desc = f"Function '{features['function_name']}'"
            
            # Add parameters with names
            if features['param_names']:
                params_str = ", ".join(features['param_names'][:5])  # Limit to 5
                if len(features['param_names']) > 5:
                    params_str += ", ..."
                func_desc += f" with params ({params_str})"
            elif features['num_params'] > 0:
                func_desc += f" with {features['num_params']} params"
            
            summary_parts.append(func_desc)
        elif features['num_functions'] > 0:
            summary_parts.append(f"{features['num_functions']} function(s)")
        
        # Control flow
        control_flow = []
        if features['num_if'] > 0:
            control_flow.append(f"{features['num_if']} conditional(s)")
        if features['num_loops'] > 0:
            control_flow.append(f"{features['num_loops']} loop(s)")
        if features['num_try'] > 0:
            control_flow.append(f"{features['num_try']} try-except(s)")
        
        if control_flow:
            summary_parts.append("has " + ", ".join(control_flow))
        
        # Called functions
        if features['called_functions']:
            calls_str = ", ".join(features['called_functions'][:5])
            if len(features['called_functions']) > 5:
                calls_str += ", ..."
            summary_parts.append(f"calls [{calls_str}]")
        elif features['num_calls'] > 0:
            summary_parts.append(f"calls {features['num_calls']} function(s)")
        
        # Returns with type if available
        if features['return_type']:
            summary_parts.append(f"returns {features['return_type']}")
        elif features['has_return']:
            summary_parts.append("returns value")
        
        if not summary_parts:
            return "Structure: simple code block"
        
        return "Structure: " + ", ".join(summary_parts)
    
    def _extract_enhanced_features(self, tree: ast.AST) -> Dict:
        """Extract enhanced structural features including names."""
        features = {
            'num_functions': 0,
            'function_name': None,
            'num_params': 0,
            'param_names': [],
            'num_if': 0,
            'num_loops': 0,
            'num_try': 0,
            'has_return': False,
            'return_type': None,
            'num_calls': 0,
            'called_functions': [],
            'num_classes': 0
        }
        
        # Find first function definition for detailed info
        first_func = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and first_func is None:
                first_func = node
                features['function_name'] = node.name
                features['num_params'] = len(node.args.args)
                
                # Extract parameter names
                features['param_names'] = [arg.arg for arg in node.args.args]
                
                # Extract return type annotation if available
                if node.returns:
                    features['return_type'] = self._get_type_name(node.returns)
                
                break
        
        # Count all features
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                features['num_functions'] += 1
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
                # Extract called function name
                func_name = self._get_call_name(node)
                if func_name and func_name not in features['called_functions']:
                    features['called_functions'].append(func_name)
        
        return features
    
    def _get_type_name(self, node: ast.AST) -> str:
        """Extract type name from annotation node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Constant):
            return str(node.value)
        elif isinstance(node, ast.Subscript):
            # For types like List[int], Dict[str, int]
            if isinstance(node.value, ast.Name):
                return node.value.id
        return None
    
    def _get_call_name(self, node: ast.Call) -> str:
        """Extract function name from Call node."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            # For method calls like obj.method() or module.func()
            if isinstance(node.func.value, ast.Name):
                return f"{node.func.value.id}.{node.func.attr}"
            return node.func.attr
        return None
