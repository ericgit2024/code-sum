"""
Execution Tracer for capturing runtime behavior of code snippets.

This module provides safe execution tracing capabilities for code summarization,
capturing function calls, variable states, control flow, and return values.
"""

import sys
import ast
import signal
import traceback
from typing import Dict, List, Any, Optional, Tuple
from contextlib import contextmanager
import io
import copy


class ExecutionTimeout(Exception):
    """Raised when code execution exceeds timeout."""
    pass


class ExecutionError(Exception):
    """Raised when code execution fails."""
    pass


def timeout_handler(signum, frame):
    """Signal handler for execution timeout."""
    raise ExecutionTimeout("Execution exceeded timeout limit")


class ExecutionTracer:
    """
    Traces code execution to capture runtime behavior.
    
    Features:
    - Safe execution with timeout
    - Captures function calls, branches, and variable states
    - Sandboxed environment with restricted builtins
    - Detailed trace information for summarization
    """
    
    def __init__(self, timeout: int = 2, max_trace_depth: int = 50):
        """
        Initialize execution tracer.
        
        Args:
            timeout: Maximum execution time in seconds
            max_trace_depth: Maximum depth of trace to capture
        """
        self.timeout = timeout
        self.max_trace_depth = max_trace_depth
        self.trace_events = []
        self.current_depth = 0
        
        # Safe builtins (restricted set)
        self.safe_builtins = {
            'abs': abs, 'all': all, 'any': any, 'bool': bool,
            'dict': dict, 'enumerate': enumerate, 'filter': filter,
            'float': float, 'int': int, 'isinstance': isinstance,
            'len': len, 'list': list, 'map': map, 'max': max,
            'min': min, 'range': range, 'reversed': reversed,
            'round': round, 'set': set, 'sorted': sorted,
            'str': str, 'sum': sum, 'tuple': tuple, 'zip': zip,
            'True': True, 'False': False, 'None': None,
        }
        
        # Safe modules (allow import of common safe modules)
        import math
        self.safe_modules = {
            'math': math
        }
    
    def trace_function(self, frame, event, arg):
        """
        Trace function for sys.settrace.
        
        Args:
            frame: Current stack frame
            event: Event type ('call', 'line', 'return', etc.)
            arg: Event argument
            
        Returns:
            Trace function for next event
        """
        if self.current_depth >= self.max_trace_depth:
            return None
        
        # Get current function info
        code = frame.f_code
        func_name = code.co_name
        line_no = frame.f_lineno
        
        # Skip internal functions
        if func_name.startswith('_') and func_name != '__init__':
            return None
        
        if event == 'call':
            self.current_depth += 1
            # Capture function call
            local_vars = {k: self._safe_repr(v) for k, v in frame.f_locals.items()}
            self.trace_events.append({
                'type': 'call',
                'function': func_name,
                'line': line_no,
                'args': local_vars,
                'depth': self.current_depth
            })
            
        elif event == 'line':
            # Capture line execution (for control flow)
            self.trace_events.append({
                'type': 'line',
                'function': func_name,
                'line': line_no,
                'depth': self.current_depth
            })
            
        elif event == 'return':
            # Capture return value
            self.trace_events.append({
                'type': 'return',
                'function': func_name,
                'line': line_no,
                'value': self._safe_repr(arg),
                'depth': self.current_depth
            })
            self.current_depth -= 1
            
        elif event == 'exception':
            # Capture exception
            exc_type, exc_value, _ = arg
            self.trace_events.append({
                'type': 'exception',
                'function': func_name,
                'line': line_no,
                'exception': f"{exc_type.__name__}: {exc_value}",
                'depth': self.current_depth
            })
        
        return self.trace_function
    
    def _safe_repr(self, obj: Any, max_length: int = 100) -> str:
        """
        Safely represent an object as string.
        
        Args:
            obj: Object to represent
            max_length: Maximum string length
            
        Returns:
            String representation
        """
        try:
            repr_str = repr(obj)
            if len(repr_str) > max_length:
                return repr_str[:max_length] + '...'
            return repr_str
        except Exception:
            return f"<{type(obj).__name__}>"
    
    @contextmanager
    def timeout_context(self):
        """Context manager for execution timeout."""
        # Set timeout alarm (Unix-like systems)
        if hasattr(signal, 'SIGALRM'):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.timeout)
        
        try:
            yield
        finally:
            # Cancel alarm
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
    
    def execute_with_trace(self, code: str, test_inputs: List[Dict[str, Any]]) -> Dict:
        """
        Execute code with multiple test inputs and capture traces.
        
        Args:
            code: Python code to execute
            test_inputs: List of test input dictionaries
            
        Returns:
            Dictionary with execution results and traces
        """
        results = {
            'success': False,
            'executions': [],
            'error': None
        }
        
        # Parse code to extract function name
        try:
            tree = ast.parse(code)
            func_def = None
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_def = node
                    break
            
            if not func_def:
                results['error'] = "No function definition found in code"
                return results
            
            func_name = func_def.name
            
        except SyntaxError as e:
            results['error'] = f"Syntax error: {e}"
            return results
        
        # Execute with each test input
        for i, test_input in enumerate(test_inputs):
            execution_result = self._execute_single(code, func_name, test_input)
            results['executions'].append(execution_result)
            
            # Stop if we have enough successful executions
            if len([e for e in results['executions'] if e['success']]) >= 3:
                break
        
        # Mark overall success if at least one execution succeeded
        results['success'] = any(e['success'] for e in results['executions'])
        
        return results
    
    def _execute_single(self, code: str, func_name: str, test_input: Dict[str, Any]) -> Dict:
        """
        Execute code with a single test input.
        
        Args:
            code: Python code to execute
            func_name: Name of function to call
            test_input: Test input arguments
            
        Returns:
            Execution result dictionary
        """
        result = {
            'success': False,
            'input': test_input,
            'output': None,
            'trace': [],
            'error': None,
            'branches_taken': [],
            'functions_called': []
        }
        
        # Reset trace
        self.trace_events = []
        self.current_depth = 0
        
        # Create sandboxed namespace
        namespace = {
            '__builtins__': self.safe_builtins,
            '__name__': '__main__',
        }
        
        # Add safe modules to namespace
        namespace.update(self.safe_modules)
        
        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        try:
            # Execute code definition
            exec(code, namespace)
            
            # Check if function exists
            if func_name not in namespace:
                result['error'] = f"Function '{func_name}' not found after execution"
                return result
            
            func = namespace[func_name]
            
            # Set trace
            sys.settrace(self.trace_function)
            
            # Execute function with timeout
            try:
                with self.timeout_context():
                    output = func(**test_input)
                    result['output'] = self._safe_repr(output)
                    result['success'] = True
            except ExecutionTimeout:
                result['error'] = "Execution timeout"
            except TypeError as e:
                result['error'] = f"Type error: {e}"
            except Exception as e:
                result['error'] = f"{type(e).__name__}: {e}"
            
            # Unset trace
            sys.settrace(None)
            
            # Process trace events
            result['trace'] = copy.deepcopy(self.trace_events)
            result['functions_called'] = list(set([
                e['function'] for e in self.trace_events 
                if e['type'] == 'call' and e['function'] != func_name
            ]))
            
            # Identify branches (simplified: count unique lines executed)
            lines_executed = set([e['line'] for e in self.trace_events if e['type'] == 'line'])
            result['branches_taken'] = sorted(list(lines_executed))
            
        except Exception as e:
            result['error'] = f"Execution error: {traceback.format_exc()}"
        
        finally:
            # Restore stdout
            sys.stdout = old_stdout
            sys.settrace(None)
        
        return result


class TestInputGenerator:
    """
    Generates test inputs for functions based on their signatures.
    """
    
    def __init__(self):
        """Initialize test input generator."""
        self.type_samples = {
            'int': [0, 1, -1, 10, 100],
            'float': [0.0, 1.0, -1.0, 3.14, 100.5],
            'str': ['', 'test', 'hello', 'a', 'long string here'],
            'bool': [True, False],
            'list': [[], [1], [1, 2, 3], ['a', 'b']],
            'dict': [{}, {'key': 'value'}, {'a': 1, 'b': 2}],
            'tuple': [(), (1,), (1, 2, 3)],
            'set': [set(), {1}, {1, 2, 3}],
        }
    
    def generate_inputs(self, code: str, max_combinations: int = 5) -> List[Dict[str, Any]]:
        """
        Generate test inputs for a function.
        
        Args:
            code: Python code containing function
            max_combinations: Maximum number of input combinations
            
        Returns:
            List of test input dictionaries
        """
        try:
            tree = ast.parse(code)
            func_def = None
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_def = node
                    break
            
            if not func_def:
                return []
            
            # Extract parameter names and annotations
            params = []
            for arg in func_def.args.args:
                param_info = {
                    'name': arg.arg,
                    'type': self._infer_type(arg, func_def)
                }
                params.append(param_info)
            
            # Generate combinations
            inputs = self._generate_combinations(params, max_combinations)
            
            return inputs
            
        except Exception as e:
            print(f"Error generating inputs: {e}")
            return []
    
    def _infer_type(self, arg: ast.arg, func_def: ast.FunctionDef) -> str:
        """
        Infer parameter type from annotation or name.
        
        Args:
            arg: AST argument node
            func_def: Function definition node
            
        Returns:
            Inferred type name
        """
        # Check annotation
        if arg.annotation:
            if isinstance(arg.annotation, ast.Name):
                return arg.annotation.id.lower()
        
        # Infer from name
        name = arg.arg.lower()
        if 'num' in name or 'count' in name or 'idx' in name or 'index' in name:
            return 'int'
        elif 'name' in name or 'text' in name or 'msg' in name or 'str' in name:
            return 'str'
        elif 'list' in name or 'items' in name or 'arr' in name:
            return 'list'
        elif 'dict' in name or 'map' in name:
            return 'dict'
        elif 'flag' in name or 'is_' in name or 'has_' in name:
            return 'bool'
        
        # Default to int
        return 'int'
    
    def _generate_combinations(self, params: List[Dict], max_combinations: int) -> List[Dict[str, Any]]:
        """
        Generate parameter combinations.
        
        Args:
            params: List of parameter info dictionaries
            max_combinations: Maximum combinations to generate
            
        Returns:
            List of input dictionaries
        """
        if not params:
            return [{}]
        
        combinations = []
        
        # Generate simple combinations (not full cartesian product)
        for i in range(min(max_combinations, 5)):
            combo = {}
            for param in params:
                param_type = param['type']
                samples = self.type_samples.get(param_type, [0])
                # Use modulo to cycle through samples
                combo[param['name']] = samples[i % len(samples)]
            combinations.append(combo)
        
        return combinations
