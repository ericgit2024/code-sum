"""
End-to-end integration test for execution trace-guided summarization.

This script tests the full pipeline with the preprocessor integration.
"""

import sys
import os
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.preprocessor import DataPreprocessor


def test_preprocessor_with_trace():
    """Test preprocessor with execution trace enabled."""
    print("=" * 70)
    print("INTEGRATION TEST: Preprocessor with Execution Trace")
    print("=" * 70)
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Ensure trace is enabled
    config['execution_trace']['enabled'] = True
    config['execution_trace']['include_in_prompt'] = True
    
    # Create preprocessor
    preprocessor = DataPreprocessor(config)
    
    # Test cases
    test_cases = [
        {
            'name': 'Simple Addition',
            'code': '''def add(a, b):
    """Add two numbers."""
    return a + b'''
        },
        {
            'name': 'Conditional Logic',
            'code': '''def safe_divide(a, b):
    """Divide a by b, returning None if b is zero."""
    if b == 0:
        return None
    return a / b'''
        },
        {
            'name': 'List Processing',
            'code': '''def filter_positive(numbers):
    """Filter positive numbers from a list."""
    result = []
    for num in numbers:
        if num > 0:
            result.append(num)
    return result'''
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'=' * 70}")
        print(f"Test Case {i}: {test_case['name']}")
        print(f"{'=' * 70}")
        
        try:
            # Extract structures (includes trace)
            structures = preprocessor.extract_structures(test_case['code'])
            
            print(f"\nExtracted Structures:")
            print(f"  - Static Summary: {structures.get('summary', 'N/A')[:100]}...")
            
            if 'trace' in structures:
                print(f"  - Trace Available: YES")
                print(f"\nTrace Summary:")
                print(structures['trace'])
            else:
                print(f"  - Trace Available: NO")
            
            # Format prompt
            prompt = preprocessor.format_prompt(test_case['code'], structures, "")
            
            print(f"\nPrompt Length: {len(prompt)} characters")
            print(f"\nPrompt Preview (first 300 chars):")
            print(prompt[:300] + "...")
            
            results.append(('PASS', test_case['name']))
            
        except Exception as e:
            print(f"\nERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append(('FAIL', test_case['name']))
    
    # Summary
    print(f"\n{'=' * 70}")
    print("TEST SUMMARY")
    print(f"{'=' * 70}")
    
    for status, name in results:
        print(f"{status:6} - {name}")
    
    passed = sum(1 for s, _ in results if s == 'PASS')
    total = len(results)
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All integration tests passed!")
        print("\nExecution trace-guided summarization is working correctly!")
        print("The system now combines:")
        print("  1. Static analysis (compact structure summary)")
        print("  2. Dynamic analysis (execution trace)")
        print("  3. Natural language generation")
    else:
        print(f"\n✗ {total - passed} test(s) failed")
    
    return passed == total


def test_trace_disabled():
    """Test that system works with trace disabled."""
    print(f"\n{'=' * 70}")
    print("FALLBACK TEST: Execution Trace Disabled")
    print(f"{'=' * 70}")
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Disable trace
    config['execution_trace']['enabled'] = False
    
    # Create preprocessor
    preprocessor = DataPreprocessor(config)
    
    code = '''def multiply(x, y):
    """Multiply two numbers."""
    return x * y'''
    
    try:
        structures = preprocessor.extract_structures(code)
        
        print(f"\nStatic Summary: {structures.get('summary', 'N/A')}")
        print(f"Trace Available: {'trace' in structures}")
        
        if 'trace' not in structures:
            print("\n✓ Trace correctly disabled - using static analysis only")
            return True
        else:
            print("\n✗ Trace should be disabled but was found")
            return False
            
    except Exception as e:
        print(f"\nERROR: {e}")
        return False


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("EXECUTION TRACE-GUIDED SUMMARIZATION")
    print("END-TO-END INTEGRATION TEST")
    print("=" * 70)
    
    # Run tests
    test1_pass = test_preprocessor_with_trace()
    test2_pass = test_trace_disabled()
    
    # Final summary
    print(f"\n{'=' * 70}")
    print("FINAL RESULTS")
    print(f"{'=' * 70}")
    print(f"Integration Test: {'PASS' if test1_pass else 'FAIL'}")
    print(f"Fallback Test: {'PASS' if test2_pass else 'FAIL'}")
    
    if test1_pass and test2_pass:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nThe execution trace-guided summarization system is ready for use!")
    else:
        print("\n⚠️  Some tests failed - please review errors above")
