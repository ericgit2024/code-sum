"""
Summary Critic Agent for analyzing whether summaries adequately explain
the main functional intent of code, with special focus on key input parameters.
"""

import ast
import re
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ParameterInfo:
    """Information about a function parameter."""
    name: str
    has_default: bool
    is_positional: bool
    annotation: Optional[str] = None


@dataclass
class CriticAnalysis:
    """Result of critic analysis."""
    needs_regeneration: bool
    instruction: str
    missing_parameter_count: int
    identified_parameters: List[str]
    explained_parameters: Set[str]
    unexplained_parameters: Set[str]
    confidence_score: float


class SummaryCriticAgent:
    """
    Analyzes whether summaries adequately explain the main functional intent
    of code, focusing on key input parameters that materially affect behavior.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the summary critic agent.
        
        Args:
            config: Configuration dictionary with optional settings:
                - ignore_self: Whether to ignore 'self' parameter (default: True)
                - ignore_cls: Whether to ignore 'cls' parameter (default: True)
                - ignore_args: Whether to ignore *args (default: True)
                - ignore_kwargs: Whether to ignore **kwargs (default: True)
                - min_explanation_threshold: Minimum ratio of explained params (default: 0.8)
        """
        self.config = config or {}
        self.ignore_self = self.config.get('ignore_self', True)
        self.ignore_cls = self.config.get('ignore_cls', True)
        self.ignore_args = self.config.get('ignore_args', True)
        self.ignore_kwargs = self.config.get('ignore_kwargs', True)
        self.min_explanation_threshold = self.config.get('min_explanation_threshold', 0.8)
    
    def analyze(self, code: str, summary: str, structural_info: Dict = None) -> CriticAnalysis:
        """
        Analyze whether the summary adequately explains the code's main functional intent.
        
        Args:
            code: Source code to analyze
            summary: Generated natural-language summary/docstring
            structural_info: Optional pre-extracted structural information from AST
            
        Returns:
            CriticAnalysis object with analysis results and optional instruction
        """
        # Extract main parameters from code
        main_parameters = self._extract_main_parameters(code, structural_info)
        
        if not main_parameters:
            # No parameters to check, summary is acceptable
            return CriticAnalysis(
                needs_regeneration=False,
                instruction="",
                missing_parameter_count=0,
                identified_parameters=[],
                explained_parameters=set(),
                unexplained_parameters=set(),
                confidence_score=1.0
            )
        
        # Analyze which parameters are explained in the summary
        explained_params = self._find_explained_parameters(summary, main_parameters)
        unexplained_params = set(p.name for p in main_parameters) - explained_params
        
        # Calculate explanation ratio
        explanation_ratio = len(explained_params) / len(main_parameters) if main_parameters else 1.0
        
        # Determine if regeneration is needed
        needs_regeneration = explanation_ratio < self.min_explanation_threshold
        
        # Generate instruction if needed
        instruction = ""
        if needs_regeneration:
            instruction = self._generate_instruction(
                main_parameters, 
                explained_params, 
                unexplained_params,
                summary
            )
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence(
            main_parameters,
            explained_params,
            summary
        )
        
        return CriticAnalysis(
            needs_regeneration=needs_regeneration,
            instruction=instruction,
            missing_parameter_count=len(unexplained_params),
            identified_parameters=[p.name for p in main_parameters],
            explained_parameters=explained_params,
            unexplained_parameters=unexplained_params,
            confidence_score=confidence_score
        )
    
    def _extract_main_parameters(self, code: str, structural_info: Dict = None) -> List[ParameterInfo]:
        """
        Extract main parameters that materially affect function behavior.
        Filters out implicit/auxiliary parameters like self, cls, *args, **kwargs.
        
        Args:
            code: Source code
            structural_info: Optional pre-extracted structural information
            
        Returns:
            List of ParameterInfo objects for main parameters
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []
        
        # Find the main function definition
        function_def = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_def = node
                break
        
        if not function_def:
            return []
        
        main_params = []
        args = function_def.args
        
        # Process regular arguments
        for i, arg in enumerate(args.args):
            param_name = arg.arg
            
            # Skip implicit/auxiliary parameters
            if self._should_ignore_parameter(param_name):
                continue
            
            # Extract annotation if present
            annotation = None
            if arg.annotation:
                annotation = ast.unparse(arg.annotation)
            
            # Check if has default value
            defaults_offset = len(args.args) - len(args.defaults)
            has_default = i >= defaults_offset
            
            main_params.append(ParameterInfo(
                name=param_name,
                has_default=has_default,
                is_positional=True,
                annotation=annotation
            ))
        
        # Process keyword-only arguments
        for i, arg in enumerate(args.kwonlyargs):
            param_name = arg.arg
            
            if self._should_ignore_parameter(param_name):
                continue
            
            annotation = None
            if arg.annotation:
                annotation = ast.unparse(arg.annotation)
            
            has_default = i < len(args.kw_defaults) and args.kw_defaults[i] is not None
            
            main_params.append(ParameterInfo(
                name=param_name,
                has_default=has_default,
                is_positional=False,
                annotation=annotation
            ))
        
        return main_params
    
    def _should_ignore_parameter(self, param_name: str) -> bool:
        """
        Determine if a parameter should be ignored as auxiliary/implicit.
        
        Args:
            param_name: Parameter name to check
            
        Returns:
            True if parameter should be ignored
        """
        if self.ignore_self and param_name == 'self':
            return True
        if self.ignore_cls and param_name == 'cls':
            return True
        if self.ignore_args and param_name.startswith('*') and not param_name.startswith('**'):
            return True
        if self.ignore_kwargs and param_name.startswith('**'):
            return True
        return False
    
    def _find_explained_parameters(self, summary: str, parameters: List[ParameterInfo]) -> Set[str]:
        """
        Find which parameters are naturally explained in the summary.
        Uses fuzzy matching to detect parameter mentions and contextual explanations.
        
        Args:
            summary: The generated summary text
            parameters: List of main parameters to check
            
        Returns:
            Set of parameter names that are explained
        """
        explained = set()
        summary_lower = summary.lower()
        
        for param in parameters:
            param_name = param.name.lower()
            
            # Direct mention patterns
            direct_patterns = [
                rf'\b{re.escape(param_name)}\b',  # Exact word match
                rf'`{re.escape(param_name)}`',     # Backtick quoted
                rf"'{re.escape(param_name)}'",     # Single quoted
                rf'"{re.escape(param_name)}"',     # Double quoted
            ]
            
            # Check for direct mentions
            for pattern in direct_patterns:
                if re.search(pattern, summary_lower):
                    explained.add(param.name)
                    break
            
            if param.name in explained:
                continue
            
            # Contextual explanation patterns (parameter role/effect described)
            # Look for phrases that indicate the parameter's influence
            contextual_patterns = [
                rf'\b(?:based on|using|with|given|from|for|by)\s+(?:the\s+)?{re.escape(param_name)}',
                rf'{re.escape(param_name)}\s+(?:determines|controls|specifies|defines|sets|indicates)',
                rf'(?:if|when|where)\s+{re.escape(param_name)}',
                rf'{re.escape(param_name)}\s+(?:is|are)\s+(?:used|provided|specified)',
            ]
            
            for pattern in contextual_patterns:
                if re.search(pattern, summary_lower):
                    explained.add(param.name)
                    break
            
            # Check for semantic variations (common parameter name patterns)
            if not param.name in explained:
                # Handle common variations like "file_path" -> "file path" or "filepath"
                variations = self._generate_parameter_variations(param_name)
                for variation in variations:
                    if variation in summary_lower:
                        explained.add(param.name)
                        break
        
        return explained
    
    def _generate_parameter_variations(self, param_name: str) -> List[str]:
        """
        Generate common variations of a parameter name.
        
        Args:
            param_name: Original parameter name
            
        Returns:
            List of possible variations
        """
        variations = [param_name]
        
        # Add space-separated version (e.g., "file_path" -> "file path")
        if '_' in param_name:
            variations.append(param_name.replace('_', ' '))
            variations.append(param_name.replace('_', ''))
        
        # Add camelCase variations
        if '_' in param_name:
            parts = param_name.split('_')
            camel = parts[0] + ''.join(p.capitalize() for p in parts[1:])
            variations.append(camel)
        
        return variations
    
    def _generate_instruction(self, main_parameters: List[ParameterInfo], 
                             explained_params: Set[str],
                             unexplained_params: Set[str],
                             current_summary: str) -> str:
        """
        Generate a natural-language instruction for summary regeneration.
        
        Args:
            main_parameters: List of main parameters
            explained_params: Set of parameters already explained
            unexplained_params: Set of parameters not explained
            current_summary: Current summary text
            
        Returns:
            Single-sentence instruction for regeneration
        """
        # Determine what conceptual information is missing
        if len(unexplained_params) == len(main_parameters):
            # No parameters explained at all
            return "Refine the description to naturally convey how the primary inputs influence the function's behavior and outcome."
        
        elif len(unexplained_params) > len(main_parameters) / 2:
            # Most parameters missing
            return "Enhance the explanation to include how the key inputs shape the function's operation and results."
        
        else:
            # Some parameters missing
            return "Expand the description to clarify how all significant inputs affect the function's behavior."
    
    def _calculate_confidence(self, main_parameters: List[ParameterInfo],
                             explained_params: Set[str],
                             summary: str) -> float:
        """
        Calculate confidence score for the analysis.
        
        Args:
            main_parameters: List of main parameters
            explained_params: Set of explained parameters
            summary: Summary text
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not main_parameters:
            return 1.0  # High confidence when no parameters to check
        
        # Base confidence on explanation ratio
        explanation_ratio = len(explained_params) / len(main_parameters)
        
        # Adjust based on summary quality indicators
        summary_length = len(summary.split())
        
        # Penalize very short summaries (likely incomplete)
        if summary_length < 10:
            explanation_ratio *= 0.7
        
        # Penalize very long summaries (might be verbose/unfocused)
        if summary_length > 100:
            explanation_ratio *= 0.9
        
        return min(1.0, explanation_ratio)
    
    def get_analysis_report(self, analysis: CriticAnalysis) -> str:
        """
        Generate a human-readable analysis report.
        
        Args:
            analysis: CriticAnalysis result
            
        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=== Summary Critic Analysis ===")
        lines.append(f"Needs Regeneration: {analysis.needs_regeneration}")
        lines.append(f"Confidence Score: {analysis.confidence_score:.2f}")
        lines.append(f"Total Main Parameters: {len(analysis.identified_parameters)}")
        lines.append(f"Explained Parameters: {len(analysis.explained_parameters)}")
        lines.append(f"Unexplained Parameters: {len(analysis.unexplained_parameters)}")
        
        if analysis.identified_parameters:
            lines.append(f"\nIdentified Parameters: {', '.join(analysis.identified_parameters)}")
        
        if analysis.explained_parameters:
            lines.append(f"Explained: {', '.join(sorted(analysis.explained_parameters))}")
        
        if analysis.unexplained_parameters:
            lines.append(f"Unexplained: {', '.join(sorted(analysis.unexplained_parameters))}")
        
        if analysis.instruction:
            lines.append(f"\nInstruction: {analysis.instruction}")
        
        return "\n".join(lines)
