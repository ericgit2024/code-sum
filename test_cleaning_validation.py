"""
Test the enhanced cleaning and validation logic.
"""

from src.agent.critic_refinement_agent import CriticRefinementAgent

# Mock config
config = {
    'summary_critic': {
        'max_iterations': 3,
        'temperature': 0.7,
        'max_tokens_refinement': 300,
        'fast_mode': False,
        'greedy_decoding': False
    }
}

# Create agent (without model for testing cleaning only)
class MockAgent:
    def _clean_summary(self, text):
        # Copy the cleaning logic from CriticRefinementAgent
        import re
        
        # Remove triple quotes at start/end
        text = text.strip()
        if text.startswith('"""') or text.startswith("'''"):
            text = text[3:]
        if text.endswith('"""') or text.endswith("'''"):
            text = text[:-3]
        text = text.strip()
        
        # Remove docstring parameter syntax lines
        text = re.sub(r':param\s+\w+:.*?(?=:param|:type|:return|:rtype|$)', '', text, flags=re.DOTALL)
        text = re.sub(r':type\s+\w+:.*?(?=:param|:type|:return|:rtype|$)', '', text, flags=re.DOTALL)
        text = re.sub(r':return:.*?(?=:param|:type|:rtype|$)', '', text, flags=re.DOTALL)
        text = re.sub(r':rtype:.*?(?=:param|:type|:return|$)', '', text, flags=re.DOTALL)
        
        # Remove "Examples:" section
        if 'Examples:' in text or 'Example:' in text:
            text = re.split(r'Examples?:', text)[0]
        
        # Remove prompt markers
        prompt_markers = ['Feedback:', 'Code:', 'Summary:', 'Docstring:', 'Output:', 
                         'Improved docstring:', 'Explanation:', 'Write', 'Description:',
                         'Current description:', 'Rewrite']
        for marker in prompt_markers:
            if marker in text:
                parts = text.split(marker)
                if text.strip().startswith(marker):
                    text = parts[1].strip() if len(parts) > 1 else text
                else:
                    text = parts[0].strip()
        
        # Process line by line
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Skip docstring syntax lines
            if line.startswith((':param', ':type', ':return', ':rtype', ':raises', ':note')):
                continue
            
            # Skip triple quotes
            if line in ['"""', "'''", '"""."""', "'''.''"]:
                continue
            
            # Remove bullet points
            if line.startswith(('- ', '* ', '• ')):
                line = line[2:].strip()
            
            # Skip code lines
            if line.startswith(('def ', 'class ', 'import ', 'from ', '>>>', '```')):
                continue
            
            # Skip example code
            if re.match(r'^\[.*\]$', line):
                continue
            
            cleaned_lines.append(line)
        
        # Join into paragraph
        result = ' '.join(cleaned_lines)
        
        # Remove remaining docstring artifacts
        result = result.replace('"""', '').replace("'''", '')
        
        # Remove multiple spaces
        while '  ' in result:
            result = result.replace('  ', ' ')
        
        # Remove multiple periods
        while '..' in result:
            result = result.replace('..', '.')
        
        # Limit to 4 sentences
        sentences = [s.strip() for s in result.split('.') if s.strip()]
        if len(sentences) > 4:
            sentences = sentences[:4]
        
        result = '. '.join(sentences)
        if result and not result.endswith('.'):
            result += '.'
        
        return result.strip()
    
    def _is_valid_summary(self, summary):
        if not summary or len(summary.strip()) == 0:
            return False
        
        words = summary.split()
        if len(words) < 3:
            return False
        
        # Check for docstring syntax
        docstring_indicators = [':param', ':type', ':return', ':rtype', ':raises', 
                               '"""', "'''", ':note', ':example']
        if any(indicator in summary for indicator in docstring_indicators):
            return False
        
        # Check it's not code
        code_indicators = ['def ', 'class ', 'import ', 'from ', '```', 'self.', 'config[']
        if any(indicator in summary for indicator in code_indicators):
            return False
        
        return True

agent = MockAgent()

# Test 1: Malformed output from your example
print("=" * 80)
print("TEST 1: Cleaning Malformed Docstring Syntax")
print("=" * 80)

malformed = '''"""Merge two sorted lists into one sorted list. :param list1: the first list to merge :type list1: list :param list2: the second list to merge :type list2: list :return: a list containing the merged lists :rtype: list Examples: [1, 2, 3, 4, 5, 6, 7, 8] [1, 2, 3].'''

print("\nOriginal:")
print(malformed)

cleaned = agent._clean_summary(malformed)
print("\nCleaned:")
print(cleaned)

is_valid = agent._is_valid_summary(cleaned)
print(f"\nIs Valid: {is_valid}")

# Test 2: Another malformed example
print("\n" + "=" * 80)
print("TEST 2: Cleaning Another Malformed Example")
print("=" * 80)

malformed2 = '''"""Calculate discount.
:param price: original price
:type price: float
:param discount: discount percentage
:type discount: float
:return: discounted price
:rtype: float
"""'''

print("\nOriginal:")
print(malformed2)

cleaned2 = agent._clean_summary(malformed2)
print("\nCleaned:")
print(cleaned2)

is_valid2 = agent._is_valid_summary(cleaned2)
print(f"\nIs Valid: {is_valid2}")

# Test 3: Good natural language (should pass through)
print("\n" + "=" * 80)
print("TEST 3: Good Natural Language (Should Pass)")
print("=" * 80)

good = "Merges two sorted lists into a single sorted list by comparing elements from both lists."

print("\nOriginal:")
print(good)

cleaned3 = agent._clean_summary(good)
print("\nCleaned:")
print(cleaned3)

is_valid3 = agent._is_valid_summary(cleaned3)
print(f"\nIs Valid: {is_valid3}")

# Test 4: Validation should reject docstring syntax
print("\n" + "=" * 80)
print("TEST 4: Validation Rejects Docstring Syntax")
print("=" * 80)

bad_examples = [
    '"""Merge lists"""',
    'Merge lists. :param list1: first list',
    'Returns a list. :return: merged list',
    'Process data. :type data: dict'
]

for i, bad in enumerate(bad_examples, 1):
    is_valid = agent._is_valid_summary(bad)
    print(f"{i}. '{bad}' -> Valid: {is_valid} (should be False)")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("""
The enhanced cleaning and validation logic:
1. Removes triple quotes from start/end
2. Removes :param, :type, :return, :rtype syntax
3. Removes Examples sections
4. Validates and rejects summaries with docstring syntax
5. Ensures only natural language passes through
""")
