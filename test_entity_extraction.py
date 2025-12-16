"""
Simple test to verify entity extraction improvements.
"""

import sys
import os
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.verification.entity_extractor import EntityExtractor

def test_extraction():
    """Test entity extraction from natural language docstrings."""
    
    extractor = EntityExtractor()
    
    # Test case: Natural language docstring
    docstring = "The function takes in a price and a discount percentage, and then calculates the final price."
    
    print("="*60)
    print("Testing Entity Extraction")
    print("="*60)
    print(f"\nDocstring:\n{docstring}")
    
    entities = extractor.extract_from_docstring(docstring)
    
    print(f"\nExtracted Entities:")
    print(f"  Functions: {sorted(entities.mentioned_functions)}")
    print(f"  Parameters: {sorted(entities.mentioned_parameters)}")
    print(f"  Variables: {sorted(entities.mentioned_variables)}")
    print(f"  Returns: {sorted(entities.mentioned_returns)}")
    
    # Check if key entities were found
    print(f"\n✓ Checks:")
    print(f"  'price' in parameters: {'price' in entities.mentioned_parameters}")
    print(f"  'discount' in parameters: {'discount' in entities.mentioned_parameters}")
    print(f"  'calculates' in functions: {'calculates' in entities.mentioned_functions}")
    
    # Test code extraction
    code = """
def calculate_discount(price: float, discount_percent: int, is_member: bool = False) -> float:
    if price < 0 or discount_percent < 0 or discount_percent > 100:
        raise ValueError("Invalid price or discount percentage")
    
    discount_amount = price * (discount_percent / 100)
    final_price = price - discount_amount
    
    if is_member:
        final_price = final_price * 0.95
    
    return round(final_price, 2)
"""
    
    print(f"\n{'='*60}")
    print("Code Entities:")
    print("="*60)
    
    code_entities = extractor.extract_from_code(code)
    print(f"  Function name: {code_entities.function_name}")
    print(f"  Parameters: {code_entities.parameters}")
    print(f"  Called functions: {sorted(code_entities.called_functions)}")
    print(f"  Variables: {sorted(code_entities.variables)}")
    print(f"  Return type: {code_entities.return_type}")

if __name__ == "__main__":
    test_extraction()
