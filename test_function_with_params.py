def find_max_value(numbers: list, default: int = 0) -> int:
    """Find the maximum value in a list of numbers."""
    if not numbers:
        return default
    
    max_val = numbers[0]
    for num in numbers[1:]:
        if num > max_val:
            max_val = num
    
    return max_val
