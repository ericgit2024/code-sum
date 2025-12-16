"""
Test cleaning of structured docstring format.
"""

import re

def clean_summary(text: str) -> str:
    """Clean generated summary by removing structured sections."""
    
    # Remove triple quotes
    text = text.strip()
    if text.startswith('"""') or text.startswith("'''"):
        text = text[3:]
    if text.endswith('"""') or text.endswith("'''"):
        text = text[:-3]
    text = text.strip()
    
    # Extract only text BEFORE structured sections
    structured_sections = [
        'Args:', 'Arguments:', 'Parameters:', 'Params:',
        'Returns:', 'Return:', 'Yields:', 'Yield:',
        'Raises:', 'Raise:', 'Throws:', 'Throw:',
        'Examples:', 'Example:', 'Usage:',
        'Notes:', 'Note:', 'Warnings:', 'Warning:',
        'Assumptions:', 'Assumption:',
        'Attributes:', 'Attribute:',
        'See Also:', 'References:'
    ]
    
    # Find earliest occurrence
    earliest_pos = len(text)
    for section in structured_sections:
        pos = text.find(section)
        if pos != -1 and pos < earliest_pos:
            earliest_pos = pos
    
    # Keep only text before structured sections
    if earliest_pos < len(text):
        text = text[:earliest_pos]
    
    # Remove docstring syntax
    text = re.sub(r':param\s+\w+:.*?(?=:param|:type|:return|:rtype|$)', '', text, flags=re.DOTALL)
    text = re.sub(r':type\s+\w+:.*?(?=:param|:type|:return|:rtype|$)', '', text, flags=re.DOTALL)
    text = re.sub(r':return:.*?(?=:param|:type|:rtype|$)', '', text, flags=re.DOTALL)
    text = re.sub(r':rtype:.*?(?=:param|:type|:return|$)', '', text, flags=re.DOTALL)
    
    # Process lines
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Skip docstring syntax
        if line.startswith((':param', ':type', ':return', ':rtype', ':raises', ':note')):
            continue
        
        # Skip section headers
        if any(line.startswith(section) for section in structured_sections):
            continue
        
        # Skip triple quotes
        if line in ['"""', "'''", '"""."""', "'''.''"]:
            continue
        
        # Remove bullet points
        if line.startswith(('- ', '* ', '• ')):
            line = line[2:].strip()
        
        # Skip code
        if line.startswith(('def ', 'class ', 'import ', 'from ', '>>>', '```')):
            continue
        
        # Skip example code
        if re.match(r'^\[.*\]$', line):
            continue
        
        # Skip lines with code (= and {})
        if '=' in line and '{' in line:
            continue
        
        cleaned_lines.append(line)
    
    # Join
    result = ' '.join(cleaned_lines)
    
    # Remove artifacts
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


# Test with your example
structured_output = '''Process a payment transaction with optional tax calculation. Args: amount: currency: apply_tax: Returns: transaction = { "amount": round(amount, 2), "currency": currency, "tax": round(tax_amount, 2), "total": round(total, 2), "taxed": apply_tax } Raises: ValueError: amount <= 0 ValueError: currency not in ["USD", "EUR", "GBP"] Assumptions: - amount <= 0, raise ValueError('Amount must be positive') - currency in ['USD', 'EUR', 'GBP'] Notes: - tax_amount = amount * tax_rate - total = amount + tax_amount Examples: - amount = 10.0, currency = 'USD', apply_tax = True - amount = 10.0, currency = 'USD', apply_tax = False - amount = 10.0, currency = 'USD', apply_tax = True -'''

print("=" * 80)
print("ORIGINAL (Structured Format):")
print("=" * 80)
print(structured_output)
print()

cleaned = clean_summary(structured_output)

print("=" * 80)
print("CLEANED (Natural Language Only):")
print("=" * 80)
print(cleaned)
print()

print("=" * 80)
print("RESULT")
print("=" * 80)
print(f"Original length: {len(structured_output)} chars")
print(f"Cleaned length: {len(cleaned)} chars")
print(f"Reduction: {100 * (1 - len(cleaned)/len(structured_output)):.1f}%")
