"""
Test script for reward function.

Tests the reward function on various code/docstring pairs to verify
that rewards align with quality expectations.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.rl.reward_function import RewardFunction


def print_separator():
    """Print a visual separator."""
    print("\n" + "="*70)


def test_case(name: str, code: str, good_doc: str, bad_doc: str, reward_fn: RewardFunction):
    """
    Test a single case with good and bad docstrings.
    
    Args:
        name: Test case name
        code: Function code
        good_doc: High-quality docstring
        bad_doc: Low-quality docstring
        reward_fn: Reward function instance
    """
    print_separator()
    print(f"TEST CASE: {name}")
    print_separator()
    print(f"\nCode:\n{code}\n")
    
    # Evaluate good docstring
    good_reward, good_breakdown = reward_fn.compute_reward(code, good_doc)
    print(f"✅ GOOD DOCSTRING: \"{good_doc}\"")
    print(f"   Total Reward: {good_reward:.3f}")
    print(f"   Breakdown:")
    for component, score in good_breakdown.items():
        print(f"     - {component}: {score:.3f}")
    
    # Evaluate bad docstring
    bad_reward, bad_breakdown = reward_fn.compute_reward(code, bad_doc)
    print(f"\n❌ BAD DOCSTRING: \"{bad_doc}\"")
    print(f"   Total Reward: {bad_reward:.3f}")
    print(f"   Breakdown:")
    for component, score in bad_breakdown.items():
        print(f"     - {component}: {score:.3f}")
    
    # Show difference
    diff = good_reward - bad_reward
    print(f"\n📊 Reward Difference: {diff:.3f} (higher is better)")
    
    if diff > 0.2:
        print("   ✓ Good discrimination (difference > 0.2)")
    else:
        print("   ⚠ Weak discrimination (difference < 0.2)")


def main():
    """Run all test cases."""
    print("\n" + "="*70)
    print("REWARD FUNCTION TEST SUITE")
    print("="*70)
    
    # Initialize reward function
    reward_fn = RewardFunction()
    
    print(f"\nReward Weights:")
    for component, weight in reward_fn.weights.items():
        print(f"  - {component}: {weight:.2f}")
    
    # Test Case 1: Simple function with parameters
    test_case(
        name="Simple Addition Function",
        code="""def add(a, b):
    return a + b""",
        good_doc="Adds two numbers a and b and returns their sum.",
        bad_doc="Does addition.",
        reward_fn=reward_fn
    )
    
    # Test Case 2: Function with control flow
    test_case(
        name="Email Validation with Error Handling",
        code="""def validate_email(email):
    if "@" not in email:
        raise ValueError("Invalid email")
    return True""",
        good_doc="Validates email address. Raises ValueError if email does not contain @. Returns True if valid.",
        bad_doc="Checks email.",
        reward_fn=reward_fn
    )
    
    # Test Case 3: Function with loop
    test_case(
        name="Sum Calculation with Loop",
        code="""def calculate_sum(numbers):
    total = 0
    for num in numbers:
        total += num
    return total""",
        good_doc="Calculates the sum of all numbers in the list by iterating through each number and adding it to the total. Returns the final sum.",
        bad_doc="Sums numbers.",
        reward_fn=reward_fn
    )
    
    # Test Case 4: Function with conditional logic
    test_case(
        name="Discount Calculation with Cap",
        code="""def calculate_discount(price, discount_percent):
    if discount_percent > 50:
        discount_percent = 50
    final_price = price * (1 - discount_percent / 100)
    return final_price""",
        good_doc="Calculates the final price after applying a discount percentage. Caps the discount at 50% maximum to prevent excessive discounts. Returns the discounted price.",
        bad_doc="Calculates discount.",
        reward_fn=reward_fn
    )
    
    # Test Case 5: Hallucination test
    test_case(
        name="Hallucination Detection",
        code="""def multiply(x, y):
    return x * y""",
        good_doc="Multiplies two numbers x and y and returns the product.",
        bad_doc="Multiplies numbers x, y, and z using the multiply_helper function and returns the result.",  # Mentions non-existent 'z' and 'multiply_helper'
        reward_fn=reward_fn
    )
    
    # Test Case 6: Code syntax in docstring
    test_case(
        name="Naturalness Check",
        code="""def is_positive(value):
    return value > 0""",
        good_doc="Checks if the value is positive and returns True if it is, False otherwise.",
        bad_doc="Checks if value > 0 and returns True/False.",  # Contains code syntax
        reward_fn=reward_fn
    )
    
    # Test Case 7: No parameters, no return
    test_case(
        name="Simple Print Function",
        code="""def print_hello():
    print("Hello, World!")""",
        good_doc="Prints a greeting message to the console.",
        bad_doc="Prints hello and returns the message.",  # Incorrectly mentions return
        reward_fn=reward_fn
    )
    
    # Summary
    print_separator()
    print("TEST SUITE COMPLETE")
    print_separator()
    print("\n✅ All test cases executed successfully!")
    print("\nNext Steps:")
    print("1. Review the reward scores to ensure they align with quality expectations")
    print("2. Adjust reward weights in config.yaml if needed")
    print("3. Run analyze_rewards.py on your test set to see overall distribution")
    print("4. Proceed with RL training using train_rl_phase1.py")


if __name__ == "__main__":
    main()
