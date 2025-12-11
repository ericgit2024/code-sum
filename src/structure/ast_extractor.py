"""
Abstract Syntax Tree (AST) extraction and encoding.
"""

import ast
from typing import Dict, List, Optional


class ASTExtractor:
    """Extracts and encodes AST from Python code."""
    
    def __init__(self, max_depth: int = 10):
        """
        Initialize AST extractor.
        
        Args:
            max_depth: Maximum depth to traverse in AST
        """
        self.max_depth = max_depth
        
    def extract(self, code: str) -> Optional[ast.AST]:
        """
        Parse code and extract AST.
        
        Args:
            code: Python source code
            
        Returns:
            AST root node or None if parsing fails
        """
        try:
            tree = ast.parse(code)
            return tree
        except SyntaxError as e:
            print(f"Syntax error in code: {e}")
            return None
        except Exception as e:
            print(f"Error parsing code: {e}")
            return None
    
    def encode_node(self, node: ast.AST, depth: int = 0) -> str:
        """
        Encode AST node to linearized string representation.
        
        Args:
            node: AST node
            depth: Current depth in tree
            
        Returns:
            Linearized string representation
        """
        if depth > self.max_depth:
            return ""
        
        node_type = node.__class__.__name__
        result = f"<{node_type}>"
        
        # Add node-specific information
        if isinstance(node, ast.Name):
            result += f"[id={node.id}]"
        elif isinstance(node, ast.Constant):
            result += f"[value={repr(node.value)[:20]}]"
        elif isinstance(node, ast.FunctionDef):
            result += f"[name={node.name}]"
        elif isinstance(node, ast.ClassDef):
            result += f"[name={node.name}]"
        elif isinstance(node, ast.Attribute):
            result += f"[attr={node.attr}]"
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                result += f"[func={node.func.id}]"
        
        # Recursively encode children
        children = []
        for child in ast.iter_child_nodes(node):
            child_encoding = self.encode_node(child, depth + 1)
            if child_encoding:
                children.append(child_encoding)
        
        if children:
            result += " { " + " ".join(children) + " }"
        
        return result
    
    def encode(self, tree: ast.AST) -> str:
        """
        Encode entire AST to string.
        
        Args:
            tree: AST root node
            
        Returns:
            Linearized AST representation
        """
        if tree is None:
            return "AST: <empty>"
        
        encoded = self.encode_node(tree)
        return f"AST: {encoded}"
    
    def extract_and_encode(self, code: str) -> str:
        """
        Extract and encode AST in one step.
        
        Args:
            code: Python source code
            
        Returns:
            Linearized AST representation
        """
        tree = self.extract(code)
        return self.encode(tree)
    
    def get_ast_features(self, code: str) -> Dict:
        """
        Extract high-level AST features.
        
        Args:
            code: Python source code
            
        Returns:
            Dictionary of AST features
        """
        tree = self.extract(code)
        if tree is None:
            return {
                'num_functions': 0,
                'num_classes': 0,
                'num_imports': 0,
                'num_calls': 0,
                'max_depth': 0
            }
        
        features = {
            'num_functions': sum(1 for _ in ast.walk(tree) if isinstance(_, ast.FunctionDef)),
            'num_classes': sum(1 for _ in ast.walk(tree) if isinstance(_, ast.ClassDef)),
            'num_imports': sum(1 for _ in ast.walk(tree) if isinstance(_, (ast.Import, ast.ImportFrom))),
            'num_calls': sum(1 for _ in ast.walk(tree) if isinstance(_, ast.Call)),
            'max_depth': self._get_max_depth(tree)
        }
        
        return features
    
    def _get_max_depth(self, node: ast.AST, current_depth: int = 0) -> int:
        """Calculate maximum depth of AST."""
        max_child_depth = current_depth
        for child in ast.iter_child_nodes(node):
            child_depth = self._get_max_depth(child, current_depth + 1)
            max_child_depth = max(max_child_depth, child_depth)
        return max_child_depth
