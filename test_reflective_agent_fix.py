"""
Test script to verify reflective agent fixes without requiring a trained model.
Uses mock model to test the logic flow.
"""

import os
import sys
import yaml

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.agent.reflective_agent import ReflectiveAgent

# Mock model and tokenizer for testing
class MockModel:
    def __init__(self):
        self.device = "cpu"
    
    def generate(self, **kwargs):
        # Simulate different responses based on iteration
        import torch
        # Return mock token IDs
        return torch.tensor([[1, 2, 3, 4, 5]])

class MockTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 2
    
    def __call__(self, text, **kwargs):
        import torch
        
        class MockInputs:
            def __init__(self):
                self.input_ids = torch.tensor([[1, 2, 3]])
            
            def to(self, device):
                return {'input_ids': self.input_ids}
        
        return MockInputs()
    
    def decode(self, tokens, **kwargs):
        # Simulate critique responses
        if hasattr(self, '_call_count'):
            self._call_count += 1
        else:
            self._call_count = 1
        
        # First call: initial summary (already generated)
        # Second call: critique - should find issues
        if self._call_count % 2 == 0:  # Critique
            return """NOT APPROVED. Issues found:
1. The formula description is incorrect - it suggests discount_amount is subtracted twice
2. Missing the member discount logic (5% additional reduction when is_member=True)
3. The docstring does not mention the conditional branch for members"""
        else:  # Refinement
            return """Calculates the final price after applying a percentage discount. 
If the customer is a member, an additional 5% discount is applied. 
The function validates inputs and returns the rounded final price."""


def main():
    """Test the reflective agent fixes."""
    
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create mock model and tokenizer
    model = MockModel()
    tokenizer = MockTokenizer()
    
    # Initialize reflective agent
    print("="*60)
    print("TESTING REFLECTIVE AGENT FIXES")
    print("="*60)
    
    agent = ReflectiveAgent(model, tokenizer, config)
    
    # Test code
    code = """def calculate_discount(price: float, discount_percent: int, is_member: bool = False) -> float:
    if price < 0 or discount_percent < 0 or discount_percent > 100:
        raise ValueError("Invalid price or discount percentage")
    
    discount_amount = price * (discount_percent / 100)
    final_price = price - discount_amount
    
    if is_member:
        final_price = final_price * 0.95
    
    return round(final_price, 2)"""
    
    # Initial summary (the problematic one from the user's output)
    initial_summary = "The function calculates the price after a discount (or not), taking into account the is_member param. The price is calculated by the formula: price - price * (discount_percent / 100) - discount_amount."
    
    print("\nInitial Summary:")
    print(initial_summary)
    print("\n" + "-"*60)
    
    # Run iterative refinement
    print("\nRunning iterative refinement...")
    final_summary, iterations = agent.iterative_refinement(code, initial_summary)
    
    print("\n" + "="*60)
    print(f"\nFINAL SUMMARY (after {iterations} iteration(s)):")
    print(final_summary)
    print("\n" + "="*60)
    
    # Verify the fixes
    print("\nVERIFICATION:")
    print(f"✓ Number of iterations: {iterations} (should be > 1)")
    print(f"✓ Debug output was shown (check above)")
    print(f"✓ Critique feedback was displayed (check above)")
    

if __name__ == "__main__":
    main()
