"""
Test script for iteration agent with sample functions.
"""

import sys
import os
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.agent.iteration_agent import IterationAgent
from src.model.model_loader import load_model
from src.structure.compact_summarizer import CompactStructureSummarizer
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Sample test functions
TEST_FUNCTIONS = [
    {
        'name': 'simple_add',
        'code': '''def add(a, b):
    return a + b''',
        'expected_issues': ['missing_parameters', 'missing_return']
    },
    {
        'name': 'with_conditionals',
        'code': '''def max_value(a, b):
    if a > b:
        return a
    else:
        return b''',
        'expected_issues': ['missing_parameters', 'missing_control_flow']
    },
    {
        'name': 'with_loop',
        'code': '''def sum_list(numbers):
    total = 0
    for num in numbers:
        total += num
    return total''',
        'expected_issues': ['missing_parameters', 'missing_control_flow']
    },
    {
        'name': 'complex_function',
        'code': '''def process_data(data, threshold=0.5):
    """Processes data."""
    results = []
    for item in data:
        if item > threshold:
            results.append(item * 2)
        else:
            results.append(item / 2)
    return results''',
        'expected_issues': []  # Has docstring already
    }
]


def test_iteration_agent():
    """Test iteration agent with sample functions."""
    
    print("="*80)
    print("ITERATION AGENT TEST")
    print("="*80)
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load model
    print("\n[1/3] Loading model...")
    hf_token = os.getenv('HF_TOKEN')
    model, tokenizer = load_model(config, hf_token, for_training=False)
    
    # Initialize iteration agent
    print("\n[2/3] Initializing iteration agent...")
    agent = IterationAgent(model, tokenizer, config)
    
    # Initialize structure summarizer
    summarizer = CompactStructureSummarizer(config)
    
    print("\n[3/3] Testing with sample functions...")
    print("="*80)
    
    for i, test_case in enumerate(TEST_FUNCTIONS, 1):
        print(f"\n{'='*80}")
        print(f"TEST CASE {i}: {test_case['name']}")
        print(f"{'='*80}")
        
        code = test_case['code']
        print(f"\nCode:\n{code}")
        
        # Extract structure
        structure = summarizer.extract_compact_summary(code)
        print(f"\nStructure: {structure}")
        
        # Generate initial summary (mock for testing)
        initial_summary = f"Function that performs operations."
        print(f"\nInitial Summary: {initial_summary}")
        
        # Run iteration agent
        final_summary, metadata = agent.iterate_once(code, initial_summary, structure)
        
        print(f"\n{'='*40}")
        print("RESULTS")
        print(f"{'='*40}")
        print(f"Final Summary: {final_summary}")
        print(f"\nMetadata:")
        print(f"  Total Issues: {metadata['total_issues']}")
        print(f"  Refined: {metadata['refined']}")
        print(f"  Word Preservation: {metadata['word_preservation']}")
        
        if metadata['validation_issues']:
            print(f"\n  Validation Issues:")
            for category, issues in metadata['validation_issues'].items():
                if issues:
                    print(f"    - {category}: {issues}")
        
        if metadata['edit_instructions']:
            print(f"\n  Edit Instructions:")
            print(f"    {metadata['edit_instructions']}")
        
        print(f"\n{'='*80}\n")
    
    print("\n" + "="*80)
    print("TEST COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    test_iteration_agent()
