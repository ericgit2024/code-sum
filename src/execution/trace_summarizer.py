"""
Trace Analyzer and Summarizer for converting execution traces to natural language.

This module analyzes execution traces and generates natural language descriptions
of runtime behavior to enhance code summarization.
"""

from typing import Dict, List, Any
import re


class TraceAnalyzer:
    """
    Analyzes execution traces to extract meaningful patterns and behaviors.
    """
    
    def __init__(self):
        """Initialize trace analyzer."""
        pass
    
    def analyze_traces(self, execution_results: Dict) -> Dict:
        """
        Analyze execution results to extract behavioral patterns.
        
        Args:
            execution_results: Results from ExecutionTracer
            
        Returns:
            Analysis dictionary with behavioral insights
        """
        analysis = {
            'success': execution_results.get('success', False),
            'total_executions': len(execution_results.get('executions', [])),
            'successful_executions': 0,
            'failed_executions': 0,
            'behaviors': [],
            'error_patterns': [],
            'control_flow_patterns': [],
            'function_calls': set(),
            'return_patterns': [],
            'edge_cases': []
        }
        
        if not execution_results.get('executions'):
            return analysis
        
        # Analyze each execution
        for execution in execution_results['executions']:
            if execution['success']:
                analysis['successful_executions'] += 1
                self._analyze_successful_execution(execution, analysis)
            else:
                analysis['failed_executions'] += 1
                self._analyze_failed_execution(execution, analysis)
        
        # Convert sets to lists for JSON serialization
        analysis['function_calls'] = sorted(list(analysis['function_calls']))
        
        return analysis
    
    def _analyze_successful_execution(self, execution: Dict, analysis: Dict):
        """Analyze a successful execution."""
        # Track function calls
        if execution.get('functions_called'):
            analysis['function_calls'].update(execution['functions_called'])
        
        # Analyze return patterns
        if execution.get('output'):
            output = execution['output']
            analysis['return_patterns'].append({
                'input': execution['input'],
                'output': output
            })
        
        # Analyze control flow
        branches = execution.get('branches_taken', [])
        if branches:
            analysis['control_flow_patterns'].append({
                'input': execution['input'],
                'lines_executed': branches
            })
        
        # Detect behaviors from trace
        trace = execution.get('trace', [])
        behaviors = self._extract_behaviors_from_trace(trace, execution['input'], execution['output'])
        analysis['behaviors'].extend(behaviors)
    
    def _analyze_failed_execution(self, execution: Dict, analysis: Dict):
        """Analyze a failed execution."""
        error = execution.get('error', 'Unknown error')
        
        # Categorize error
        if 'timeout' in error.lower():
            error_type = 'timeout'
        elif 'type error' in error.lower():
            error_type = 'type_error'
        elif 'zero' in error.lower() or 'division' in error.lower():
            error_type = 'division_by_zero'
        elif 'index' in error.lower():
            error_type = 'index_error'
        elif 'key' in error.lower():
            error_type = 'key_error'
        else:
            error_type = 'general_error'
        
        analysis['error_patterns'].append({
            'input': execution['input'],
            'error': error,
            'type': error_type
        })
        
        # This is an edge case
        analysis['edge_cases'].append({
            'input': execution['input'],
            'behavior': f"Raises {error_type}"
        })
    
    def _extract_behaviors_from_trace(self, trace: List[Dict], input_data: Dict, output: str) -> List[str]:
        """Extract behavioral descriptions from trace events."""
        behaviors = []
        
        # Check for conditional execution
        call_events = [e for e in trace if e['type'] == 'call']
        return_events = [e for e in trace if e['type'] == 'return']
        exception_events = [e for e in trace if e['type'] == 'exception']
        
        # Behavior: Function calls other functions
        called_funcs = [e['function'] for e in call_events if e.get('depth', 0) > 1]
        if called_funcs:
            behaviors.append(f"Calls {', '.join(set(called_funcs))}")
        
        # Behavior: Exception handling
        if exception_events:
            behaviors.append("Handles exceptions")
        
        # Behavior: Returns specific type
        if output:
            if output.startswith('[') or output.startswith('('):
                behaviors.append("Returns collection")
            elif output in ['True', 'False']:
                behaviors.append("Returns boolean")
            elif output == 'None':
                behaviors.append("Returns None")
            elif output.replace('.', '').replace('-', '').isdigit():
                behaviors.append("Returns numeric value")
            else:
                behaviors.append("Returns string")
        
        return behaviors


class TraceSummarizer:
    """
    Converts trace analysis into natural language summaries.
    """
    
    def __init__(self):
        """Initialize trace summarizer."""
        self.analyzer = TraceAnalyzer()
    
    def summarize_execution(self, execution_results: Dict, code: str = "") -> str:
        """
        Generate natural language summary from execution results.
        
        Args:
            execution_results: Results from ExecutionTracer
            code: Original code (optional, for context)
            
        Returns:
            Natural language summary of runtime behavior
        """
        # Analyze traces
        analysis = self.analyzer.analyze_traces(execution_results)
        
        if not analysis['success']:
            return self._summarize_failed_execution(analysis)
        
        return self._summarize_successful_execution(analysis)
    
    def _summarize_successful_execution(self, analysis: Dict) -> str:
        """Generate summary for successful executions."""
        parts = []
        
        # Return behavior
        if analysis['return_patterns']:
            return_summary = self._summarize_return_patterns(analysis['return_patterns'])
            if return_summary:
                parts.append(return_summary)
        
        # Function calls
        if analysis['function_calls']:
            funcs = ', '.join(analysis['function_calls'][:3])  # Limit to 3
            if len(analysis['function_calls']) > 3:
                funcs += ', etc.'
            parts.append(f"Uses {funcs}")
        
        # Behaviors
        unique_behaviors = list(set(analysis['behaviors']))
        if unique_behaviors:
            parts.extend(unique_behaviors[:2])  # Limit to 2 behaviors
        
        # Edge cases
        if analysis['edge_cases']:
            edge_summary = self._summarize_edge_cases(analysis['edge_cases'])
            if edge_summary:
                parts.append(edge_summary)
        
        # Combine into summary
        if parts:
            return '. '.join(parts) + '.'
        else:
            return "Executes successfully with provided inputs."
    
    def _summarize_failed_execution(self, analysis: Dict) -> str:
        """Generate summary for failed executions."""
        if not analysis['error_patterns']:
            return "Execution failed."
        
        # Summarize error patterns
        error_types = [e['type'] for e in analysis['error_patterns']]
        unique_errors = list(set(error_types))
        
        if 'division_by_zero' in unique_errors:
            return "May raise division by zero error with certain inputs."
        elif 'type_error' in unique_errors:
            return "Requires specific input types to execute correctly."
        elif 'timeout' in unique_errors:
            return "May have infinite loop or long execution time."
        else:
            return f"May raise {unique_errors[0].replace('_', ' ')} with certain inputs."
    
    def _summarize_return_patterns(self, return_patterns: List[Dict]) -> str:
        """Summarize return value patterns."""
        if not return_patterns:
            return ""
        
        # Analyze return values
        outputs = [p['output'] for p in return_patterns]
        
        # Check if all returns are same type
        if all('None' in o for o in outputs):
            return "Returns None"
        elif all(o in ['True', 'False'] for o in outputs):
            return "Returns boolean value"
        elif all(o.replace('.', '').replace('-', '').isdigit() for o in outputs if o):
            # Check if it's a transformation
            inputs = [p['input'] for p in return_patterns]
            if self._is_transformation(inputs, outputs):
                return "Transforms input values"
            else:
                return "Computes numeric result"
        elif all(o.startswith('[') or o.startswith('(') for o in outputs if o):
            return "Returns collection of values"
        else:
            return "Returns processed result"
    
    def _is_transformation(self, inputs: List[Dict], outputs: List[str]) -> bool:
        """Check if function transforms inputs."""
        # Simple heuristic: if output contains input values
        for inp, out in zip(inputs, outputs):
            for value in inp.values():
                if str(value) in out:
                    return True
        return False
    
    def _summarize_edge_cases(self, edge_cases: List[Dict]) -> str:
        """Summarize edge case handling."""
        if not edge_cases:
            return ""
        
        # Group by behavior
        behaviors = [e['behavior'] for e in edge_cases]
        unique_behaviors = list(set(behaviors))
        
        if len(unique_behaviors) == 1:
            return unique_behaviors[0]
        else:
            return f"Handles {len(unique_behaviors)} edge cases"
    
    def generate_trace_context(self, execution_results: Dict, code: str = "") -> str:
        """
        Generate trace context for prompt augmentation.
        
        Args:
            execution_results: Results from ExecutionTracer
            code: Original code
            
        Returns:
            Formatted trace context for LLM prompt
        """
        summary = self.summarize_execution(execution_results, code)
        
        # Format for prompt
        context = f"\nRuntime Behavior Analysis:\n{summary}\n"
        
        # Add specific examples if available
        analysis = self.analyzer.analyze_traces(execution_results)
        if analysis['return_patterns']:
            context += "\nExample executions:\n"
            for i, pattern in enumerate(analysis['return_patterns'][:2]):  # Show 2 examples
                input_str = ', '.join([f"{k}={v}" for k, v in pattern['input'].items()])
                context += f"- Input({input_str}) → {pattern['output']}\n"
        
        return context
