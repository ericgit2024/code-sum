"""
Test script for execution trace-guided summarization.

This script tests the execution tracer, test input generator, and trace summarizer
to ensure they work correctly before full integration.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.execution import ExecutionTracer, TestInputGenerator, TraceSummarizer


def test_simple_function():
    """Test with a simple function."""
    print("=" * 60)
    print("TEST 1: Simple Addition Function")
    print("=" * 60)
    
    code = """
def add(a, b):
    return a + b
"""
    
    # Generate test inputs
    input_gen = TestInputGenerator()
    test_inputs = input_gen.generate_inputs(code, max_combinations=3)
    print(f"\nGenerated {len(test_inputs)} test inputs:")
    for inp in test_inputs:
        print(f"  {inp}")
    
    # Execute with tracing
    tracer = ExecutionTracer(timeout=2)
    results = tracer.execute_with_trace(code, test_inputs)
    
    print(f"\nExecution success: {results['success']}")
    print(f"Number of executions: {len(results['executions'])}")
    
    # Generate summary
    summarizer = TraceSummarizer()
    summary = summarizer.summarize_execution(results, code)
    print(f"\nTrace Summary:\n{summary}")
    
    trace_context = summarizer.generate_trace_context(results, code)
    print(f"\nTrace Context for Prompt:\n{trace_context}")
    
    return results['success']


def test_conditional_function():
    """Test with a function containing conditionals."""
    print("\n" + "=" * 60)
    print("TEST 2: Function with Conditionals")
    print("=" * 60)
    
    code = """
def divide(a, b):
    if b == 0:
        return None
    return a / b
"""
    
    # Generate test inputs
    input_gen = TestInputGenerator()
    test_inputs = input_gen.generate_inputs(code, max_combinations=5)
    print(f"\nGenerated {len(test_inputs)} test inputs:")
    for inp in test_inputs:
        print(f"  {inp}")
    
    # Execute with tracing
    tracer = ExecutionTracer(timeout=2)
    results = tracer.execute_with_trace(code, test_inputs)
    
    print(f"\nExecution success: {results['success']}")
    
    # Show execution details
    for i, exec_result in enumerate(results['executions']):
        print(f"\nExecution {i+1}:")
        print(f"  Input: {exec_result['input']}")
        print(f"  Output: {exec_result['output']}")
        print(f"  Success: {exec_result['success']}")
        if exec_result.get('error'):
            print(f"  Error: {exec_result['error']}")
    
    # Generate summary
    summarizer = TraceSummarizer()
    summary = summarizer.summarize_execution(results, code)
    print(f"\nTrace Summary:\n{summary}")
    
    return results['success']


def test_complex_function():
    """Test with a more complex function."""
    print("\n" + "=" * 60)
    print("TEST 3: Complex Function with Multiple Operations")
    print("=" * 60)
    
    code = """
def process_list(items, threshold):
    result = []
    for item in items:
        if item > threshold:
            result.append(item * 2)
    return result
"""
    
    # Generate test inputs
    input_gen = TestInputGenerator()
    test_inputs = input_gen.generate_inputs(code, max_combinations=3)
    print(f"\nGenerated {len(test_inputs)} test inputs:")
    for inp in test_inputs:
        print(f"  {inp}")
    
    # Execute with tracing
    tracer = ExecutionTracer(timeout=2)
    results = tracer.execute_with_trace(code, test_inputs)
    
    print(f"\nExecution success: {results['success']}")
    
    # Generate summary
    summarizer = TraceSummarizer()
    summary = summarizer.summarize_execution(results, code)
    print(f"\nTrace Summary:\n{summary}")
    
    trace_context = summarizer.generate_trace_context(results, code)
    print(f"\nTrace Context for Prompt:\n{trace_context}")
    
    return results['success']


def test_function_with_calls():
    """Test with a function that calls other functions."""
    print("\n" + "=" * 60)
    print("TEST 4: Function with Function Calls")
    print("=" * 60)
    
    code = """
def calculate_distance(x, y):
    import math
    return math.sqrt(x**2 + y**2)
"""
    
    # Generate test inputs
    input_gen = TestInputGenerator()
    test_inputs = input_gen.generate_inputs(code, max_combinations=3)
    print(f"\nGenerated {len(test_inputs)} test inputs:")
    for inp in test_inputs:
        print(f"  {inp}")
    
    # Execute with tracing
    tracer = ExecutionTracer(timeout=2)
    results = tracer.execute_with_trace(code, test_inputs)
    
    print(f"\nExecution success: {results['success']}")
    
    # Generate summary
    summarizer = TraceSummarizer()
    summary = summarizer.summarize_execution(results, code)
    print(f"\nTrace Summary:\n{summary}")
    
    return results['success']


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("EXECUTION TRACE-GUIDED SUMMARIZATION - TEST SUITE")
    print("=" * 60)
    
    tests = [
        test_simple_function,
        test_conditional_function,
        test_complex_function,
        test_function_with_calls
    ]
    
    results = []
    for test in tests:
        try:
            success = test()
            results.append(('PASS' if success else 'FAIL', test.__name__))
        except Exception as e:
            print(f"\nERROR in {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            results.append(('ERROR', test.__name__))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for status, name in results:
        print(f"{status:6} - {name}")
    
    passed = sum(1 for s, _ in results if s == 'PASS')
    total = len(results)
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All tests passed!")
    else:
        print(f"\n✗ {total - passed} test(s) failed")


if __name__ == "__main__":
    main()
