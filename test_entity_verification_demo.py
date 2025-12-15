"""
Demo script for entity verification module.
"""

import sys
sys.path.append('.')

from src.verification.entity_extractor import EntityExtractor
from src.verification.entity_verifier import EntityVerifier
from src.verification.instruction_agent import InstructionAgent


def test_perfect_match():
    """Test case where docstring perfectly matches code."""
    print("="*60)
    print("TEST 1: Perfect Match")
    print("="*60)
    
    code = """
def calculate_distance(x, y, z):
    result = math.sqrt(x**2 + y**2 + z**2)
    return result
"""
    
    docstring = "Calculates the Euclidean distance using parameters x, y, and z. Returns the computed distance."
    
    config = {
        'entity_verification': {
            'hallucination_threshold': 0.30,
            'require_all_params': True,
            'allow_synonyms': True
        }
    }
    
    verifier = EntityVerifier(config)
    result = verifier.verify(code, docstring)
    
    print(verifier.get_verification_summary(result))
    print()


def test_hallucination():
    """Test case with hallucinated entities."""
    print("="*60)
    print("TEST 2: Hallucination Detection")
    print("="*60)
    
    code = """
def add_numbers(a, b):
    return a + b
"""
    
    # Docstring mentions non-existent entities
    docstring = "Adds numbers a, b, and c using the calculate_sum function. Returns the total."
    
    config = {
        'entity_verification': {
            'hallucination_threshold': 0.30,
            'require_all_params': True,
            'allow_synonyms': True
        }
    }
    
    verifier = EntityVerifier(config)
    instruction_agent = InstructionAgent(config)
    
    result = verifier.verify(code, docstring)
    
    print(verifier.get_verification_summary(result))
    print()
    
    if not result.passes_threshold:
        print("INSTRUCTIONS FOR REFINEMENT:")
        print("-" * 60)
        instructions = instruction_agent.generate_instructions(result, code, docstring)
        print(instructions)
    print()


def test_missing_entities():
    """Test case with missing entities."""
    print("="*60)
    print("TEST 3: Missing Entities")
    print("="*60)
    
    code = """
def process_data(input_file, output_file, threshold):
    data = read_file(input_file)
    filtered = filter_data(data, threshold)
    write_file(output_file, filtered)
    return filtered
"""
    
    # Docstring doesn't mention all parameters
    docstring = "Processes data from input_file and writes results."
    
    config = {
        'entity_verification': {
            'hallucination_threshold': 0.30,
            'require_all_params': True,
            'allow_synonyms': True
        }
    }
    
    verifier = EntityVerifier(config)
    instruction_agent = InstructionAgent(config)
    
    result = verifier.verify(code, docstring)
    
    print(verifier.get_verification_summary(result))
    print()
    
    if not result.passes_threshold:
        print("INSTRUCTIONS FOR REFINEMENT:")
        print("-" * 60)
        instructions = instruction_agent.generate_instructions(result, code, docstring)
        print(instructions)
    print()


def test_complex_function():
    """Test case with complex function."""
    print("="*60)
    print("TEST 4: Complex Function with Mixed Issues")
    print("="*60)
    
    code = """
def merge_sorted_lists(list1, list2):
    result = []
    i, j = 0, 0
    
    while i < len(list1) and j < len(list2):
        if list1[i] < list2[j]:
            result.append(list1[i])
            i += 1
        else:
            result.append(list2[j])
            j += 1
    
    result.extend(list1[i:])
    result.extend(list2[j:])
    
    return result
"""
    
    # Docstring has both hallucinations and missing entities
    docstring = "Merges list1 and list3 using the quicksort algorithm. Returns sorted output."
    
    config = {
        'entity_verification': {
            'hallucination_threshold': 0.30,
            'require_all_params': True,
            'allow_synonyms': True
        }
    }
    
    verifier = EntityVerifier(config)
    instruction_agent = InstructionAgent(config)
    
    result = verifier.verify(code, docstring)
    
    print(verifier.get_verification_summary(result))
    print()
    
    if not result.passes_threshold:
        print("INSTRUCTIONS FOR REFINEMENT:")
        print("-" * 60)
        instructions = instruction_agent.generate_instructions(result, code, docstring)
        print(instructions)
        
        print("\n" + "="*60)
        print("REFINEMENT PROMPT:")
        print("="*60)
        prompt = instruction_agent.create_refinement_prompt(code, docstring, result)
        print(prompt)
    print()


def test_entity_extraction():
    """Test entity extraction capabilities."""
    print("="*60)
    print("TEST 5: Entity Extraction Details")
    print("="*60)
    
    code = """
def calculate_statistics(data, threshold=0.5):
    mean_val = calculate_mean(data)
    std_val = calculate_std(data)
    
    if mean_val > threshold:
        result = normalize(data)
    else:
        result = data
    
    return result
"""
    
    extractor = EntityExtractor()
    code_entities = extractor.extract_from_code(code)
    
    print("CODE ENTITIES:")
    print(f"  Function: {code_entities.function_name}")
    print(f"  Parameters: {code_entities.parameters}")
    print(f"  Variables: {code_entities.variables}")
    print(f"  Called Functions: {code_entities.called_functions}")
    print(f"  Return Type: {code_entities.return_type or 'None'}")
    print(f"  Control Flow: {code_entities.control_flow}")
    print()
    
    docstring = "Calculates statistics for the data parameter using threshold. Calls calculate_mean and normalize functions. Returns normalized result."
    
    doc_entities = extractor.extract_from_docstring(docstring)
    
    print("DOCSTRING ENTITIES:")
    print(f"  Mentioned Functions: {doc_entities.mentioned_functions}")
    print(f"  Mentioned Parameters: {doc_entities.mentioned_parameters}")
    print(f"  Mentioned Variables: {doc_entities.mentioned_variables}")
    print(f"  Mentioned Returns: {doc_entities.mentioned_returns}")
    print()


if __name__ == "__main__":
    print("\n" + "🔍 ENTITY VERIFICATION MODULE DEMO" + "\n")
    
    test_perfect_match()
    test_hallucination()
    test_missing_entities()
    test_complex_function()
    test_entity_extraction()
    
    print("="*60)
    print("✅ All tests completed!")
    print("="*60)
