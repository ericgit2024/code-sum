"""
Demo script to test the new EntityRefinementAgent.
"""

import sys
import os
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.verification.entity_verifier import EntityVerifier
from src.verification.instruction_agent import InstructionAgent

def test_entity_verification():
    """Test entity verification and instruction generation."""
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize components
    verifier = EntityVerifier(config)
    instructor = InstructionAgent(config)
    
    # Test case 1: Good docstring (should pass)
    print("="*60)
    print("TEST 1: Good Docstring (Should Pass)")
    print("="*60)
    
    code1 = """
def calculate_sum(numbers, initial_value=0):
    total = initial_value
    for num in numbers:
        total += num
    return total
"""
    
    docstring1 = "Calculate the sum of numbers starting from an initial value. Takes a list of numbers and an optional initial_value parameter, returns the total sum."
    
    result1 = verifier.verify(code1, docstring1)
    print(f"\nCode:\n{code1}")
    print(f"\nDocstring:\n{docstring1}")
    print(f"\n✓ Passes: {result1.passes_threshold}")
    print(f"  Hallucination Score: {result1.hallucination_score:.3f}")
    print(f"  Precision: {result1.precision:.3f}")
    print(f"  Recall: {result1.recall:.3f}")
    print(f"  F1 Score: {result1.f1_score:.3f}")
    
    if not result1.passes_threshold:
        feedback1 = instructor.generate_instructions(result1, code1, docstring1)
        print(f"\nFeedback:\n{feedback1}")
    
    # Test case 2: Hallucinated entities (should fail)
    print("\n" + "="*60)
    print("TEST 2: Hallucinated Entities (Should Fail)")
    print("="*60)
    
    code2 = """
def read_file(filename):
    with open(filename, 'r') as f:
        return f.read()
"""
    
    docstring2 = "Read and validate a file. Opens the file, validates the content using validate_content(), processes the data with process_data(), and returns the cleaned text."
    
    result2 = verifier.verify(code2, docstring2)
    print(f"\nCode:\n{code2}")
    print(f"\nDocstring:\n{docstring2}")
    print(f"\n✗ Passes: {result2.passes_threshold}")
    print(f"  Hallucination Score: {result2.hallucination_score:.3f}")
    print(f"  Precision: {result2.precision:.3f}")
    print(f"  Recall: {result2.recall:.3f}")
    print(f"  F1 Score: {result2.f1_score:.3f}")
    
    if result2.false_positives:
        print(f"\n  Hallucinated: {', '.join(sorted(result2.false_positives))}")
    if result2.false_negatives:
        print(f"  Missing: {', '.join(sorted(result2.false_negatives))}")
    
    if not result2.passes_threshold:
        feedback2 = instructor.generate_instructions(result2, code2, docstring2)
        print(f"\n📋 Feedback Generated:\n{feedback2}")
    
    # Test case 3: Missing parameters (should fail)
    print("\n" + "="*60)
    print("TEST 3: Missing Parameters (Should Fail)")
    print("="*60)
    
    code3 = """
def format_text(text, width, align='left', fill_char=' '):
    if align == 'left':
        return text.ljust(width, fill_char)
    elif align == 'right':
        return text.rjust(width, fill_char)
    else:
        return text.center(width, fill_char)
"""
    
    docstring3 = "Format text with alignment. Returns formatted text."
    
    result3 = verifier.verify(code3, docstring3)
    print(f"\nCode:\n{code3}")
    print(f"\nDocstring:\n{docstring3}")
    print(f"\n✗ Passes: {result3.passes_threshold}")
    print(f"  Hallucination Score: {result3.hallucination_score:.3f}")
    print(f"  Precision: {result3.precision:.3f}")
    print(f"  Recall: {result3.recall:.3f}")
    print(f"  F1 Score: {result3.f1_score:.3f}")
    print(f"  Parameter Recall: {result3.parameter_recall:.3f}")
    
    if result3.false_positives:
        print(f"\n  Hallucinated: {', '.join(sorted(result3.false_positives))}")
    if result3.false_negatives:
        print(f"  Missing: {', '.join(sorted(result3.false_negatives))}")
    
    if not result3.passes_threshold:
        feedback3 = instructor.generate_instructions(result3, code3, docstring3)
        print(f"\n📋 Feedback Generated:\n{feedback3}")
    
    print("\n" + "="*60)
    print("DEMO COMPLETE!")
    print("="*60)
    print("\nKey Observations:")
    print("1. Entity verification catches hallucinated functions")
    print("2. Instruction agent generates specific, actionable feedback")
    print("3. Missing parameters are detected and reported")
    print("4. Feedback is clear and tells the model exactly what to fix")

if __name__ == "__main__":
    test_entity_verification()
