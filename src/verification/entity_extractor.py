"""
Entity extractor for code and docstrings.
"""

import ast
import re
from typing import Dict, List, Set
from dataclasses import dataclass


@dataclass
class CodeEntities:
    """Entities extracted from code."""
    function_name: str
    parameters: List[str]
    variables: Set[str]
    called_functions: Set[str]
    return_type: str
    control_flow: Dict[str, int]  # if/else, loops, try/except counts


@dataclass
class DocstringEntities:
    """Entities extracted from docstring."""
    mentioned_functions: Set[str]
    mentioned_parameters: Set[str]
    mentioned_variables: Set[str]
    mentioned_returns: Set[str]


class EntityExtractor:
    """Extract entities from code and docstrings."""
    
    def __init__(self, config: Dict = None):
        """
        Initialize entity extractor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
    def extract_from_code(self, code: str) -> CodeEntities:
        """
        Extract entities from Python code using AST.
        
        Args:
            code: Python source code
            
        Returns:
            CodeEntities object with extracted entities
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            # Return empty entities if code is malformed
            return CodeEntities(
                function_name="",
                parameters=[],
                variables=set(),
                called_functions=set(),
                return_type="",
                control_flow={}
            )
        
        # Find the main function definition
        function_def = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_def = node
                break
        
        if not function_def:
            return CodeEntities(
                function_name="",
                parameters=[],
                variables=set(),
                called_functions=set(),
                return_type="",
                control_flow={}
            )
        
        # Extract function name
        function_name = function_def.name
        
        # Extract parameters
        parameters = [arg.arg for arg in function_def.args.args]
        
        # Extract return type annotation
        return_type = ""
        if function_def.returns:
            return_type = ast.unparse(function_def.returns)
        
        # Extract variables (assigned names)
        variables = set()
        for node in ast.walk(function_def):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        variables.add(target.id)
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name):
                    variables.add(node.target.id)
        
        # Extract called functions
        called_functions = set()
        for node in ast.walk(function_def):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    called_functions.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    # For method calls like obj.method()
                    called_functions.add(node.func.attr)
        
        # Extract control flow counts
        control_flow = {
            'conditionals': 0,
            'loops': 0,
            'try_except': 0
        }
        
        for node in ast.walk(function_def):
            if isinstance(node, ast.If):
                control_flow['conditionals'] += 1
            elif isinstance(node, (ast.For, ast.While)):
                control_flow['loops'] += 1
            elif isinstance(node, ast.Try):
                control_flow['try_except'] += 1
        
        return CodeEntities(
            function_name=function_name,
            parameters=parameters,
            variables=variables,
            called_functions=called_functions,
            return_type=return_type,
            control_flow=control_flow
        )
    
    def extract_from_docstring(self, docstring: str) -> DocstringEntities:
        """
        Extract entities from docstring using pattern matching and word-based extraction.
        
        Args:
            docstring: Generated docstring text
            
        Returns:
            DocstringEntities object with extracted entities
        """
        if not docstring:
            return DocstringEntities(
                mentioned_functions=set(),
                mentioned_parameters=set(),
                mentioned_variables=set(),
                mentioned_returns=set()
            )
        
        # Normalize docstring
        docstring_lower = docstring.lower()
        
        # Extract all potential identifier words (snake_case or camelCase)
        # This catches entities even when used in natural language
        identifier_pattern = r'\b([a-z_][a-z0-9_]*)\b'
        all_words = set(re.findall(identifier_pattern, docstring_lower))
        
        # Remove common English words that aren't entities
        common_words = {
            'the', 'a', 'an', 'and', 'or', 'if', 'when', 'that', 'this', 'is', 'are',
            'be', 'to', 'of', 'in', 'for', 'on', 'with', 'as', 'by', 'from', 'at',
            'it', 'its', 'has', 'have', 'had', 'was', 'were', 'been', 'being',
            'do', 'does', 'did', 'done', 'doing', 'will', 'would', 'should', 'could',
            'may', 'might', 'must', 'can', 'then', 'than', 'based', 'using', 'used',
            'takes', 'returns', 'return', 'calculates', 'calculate', 'applies', 'apply',
            'function', 'method', 'class', 'object', 'value', 'values', 'item', 'items',
            'also', 'all', 'each', 'every', 'any', 'some', 'no', 'not', 'only',
            'first', 'last', 'next', 'previous', 'new', 'old', 'same', 'different',
            'get', 'set', 'add', 'remove', 'delete', 'update', 'create', 'make',
            'exception', 'error', 'raises', 'raise', 'raised', 'less', 'greater',
            'percentage', 'percent', 'amount', 'number', 'count', 'total', 'sum',
            'final', 'initial', 'current', 'default', 'optional', 'required',
            'true', 'false', 'none', 'null', 'empty', 'full'
        }
        
        potential_entities = all_words - common_words
        
        # Extract mentioned functions (look for function call patterns)
        # Patterns: "calls function_name", "uses function_name", "function_name()"
        function_patterns = [
            r'calls?\\s+([a-z_][a-z0-9_]*)',
            r'uses?\\s+([a-z_][a-z0-9_]*)',
            r'invokes?\\s+([a-z_][a-z0-9_]*)',
            r'([a-z_][a-z0-9_]*)\\s*\\(\\)',  # function_name()
        ]
        
        mentioned_functions = set()
        for pattern in function_patterns:
            matches = re.findall(pattern, docstring_lower)
            mentioned_functions.update(matches)
        
        # Also include potential entities that look like function names (snake_case with verb)
        # Common function name patterns: calculate_, get_, set_, process_, validate_, etc.
        function_prefixes = ['calculate', 'get', 'set', 'process', 'validate', 'check', 
                            'update', 'create', 'delete', 'remove', 'add', 'find', 
                            'search', 'filter', 'sort', 'parse', 'format', 'convert']
        for entity in potential_entities:
            if any(entity.startswith(prefix) for prefix in function_prefixes):
                mentioned_functions.add(entity)
        
        # Extract mentioned parameters
        # Patterns: "parameter param_name", "param_name parameter", "`param_name`"
        # PLUS: any identifier that appears in natural language contexts
        param_patterns = [
            r'parameter\\s+([a-z_][a-z0-9_]*)',
            r'param\\s+([a-z_][a-z0-9_]*)',
            r'argument\\s+([a-z_][a-z0-9_]*)',
            r'`([a-z_][a-z0-9_]*)`',  # backtick-quoted names
            r"'([a-z_][a-z0-9_]*)'",  # single-quoted names
            r'takes?\\s+(?:in\\s+)?(?:a\\s+)?([a-z_][a-z0-9_]*)',  # "takes in a param_name"
            r'accepts?\\s+(?:a\\s+)?([a-z_][a-z0-9_]*)',  # "accepts param_name"
            r'receives?\\s+(?:a\\s+)?([a-z_][a-z0-9_]*)',  # "receives param_name"
        ]
        
        mentioned_parameters = set()
        for pattern in param_patterns:
            matches = re.findall(pattern, docstring_lower)
            mentioned_parameters.update(matches)
        
        # Add all potential entities as possible parameters
        # (they'll be filtered by the verifier based on actual code parameters)
        mentioned_parameters.update(potential_entities)
        
        # Extract mentioned variables (similar to parameters but in different context)
        var_patterns = [
            r'variable\\s+([a-z_][a-z0-9_]*)',
            r'stores?\\s+(?:in\\s+)?([a-z_][a-z0-9_]*)',
        ]
        
        mentioned_variables = set()
        for pattern in var_patterns:
            matches = re.findall(pattern, docstring_lower)
            mentioned_variables.update(matches)
        
        # Add potential entities as variables too
        mentioned_variables.update(potential_entities)
        
        # Extract return type mentions
        return_patterns = [
            r'returns?\\s+(?:a\\s+)?(?:the\\s+)?([a-z_][a-z0-9_]*)',
            r'outputs?\\s+(?:a\\s+)?(?:the\\s+)?([a-z_][a-z0-9_]*)',
            r'yields?\\s+(?:a\\s+)?(?:the\\s+)?([a-z_][a-z0-9_]*)',
        ]
        
        mentioned_returns = set()
        for pattern in return_patterns:
            matches = re.findall(pattern, docstring_lower)
            # Filter out common words that aren't types
            mentioned_returns.update([m for m in matches if m not in common_words])
        
        return DocstringEntities(
            mentioned_functions=mentioned_functions,
            mentioned_parameters=mentioned_parameters,
            mentioned_variables=mentioned_variables,
            mentioned_returns=mentioned_returns
        )
    
    
    def normalize_entity(self, entity: str) -> str:
        """
        Normalize entity name for comparison.
        
        Args:
            entity: Entity name
            
        Returns:
            Normalized entity name
        """
        # Convert to lowercase
        normalized = entity.lower().strip()
        
        # Remove common prefixes/suffixes
        normalized = normalized.replace('_', '')
        
        return normalized
    
    def are_synonyms(self, entity1: str, entity2: str) -> bool:
        """
        Check if two entities are synonyms.
        
        Args:
            entity1: First entity
            entity2: Second entity
            
        Returns:
            True if entities are synonyms
        """
        # Common synonyms in docstrings
        synonyms = {
            'returns': {'outputs', 'yields', 'produces', 'gives'},
            'parameter': {'param', 'argument', 'arg', 'input'},
            'variable': {'var', 'value', 'val'},
        }
        
        entity1_norm = entity1.lower()
        entity2_norm = entity2.lower()
        
        # Check if they're in the same synonym group
        for key, syn_set in synonyms.items():
            if entity1_norm in syn_set and entity2_norm in syn_set:
                return True
            if entity1_norm == key and entity2_norm in syn_set:
                return True
            if entity2_norm == key and entity1_norm in syn_set:
                return True
        
        return False
