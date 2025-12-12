def merge_sorted_lists(list1: list, list2: list) -> list:
    """Merge two sorted lists into one sorted list."""
    if not list1:
        return list2
    if not list2:
        return list1
    
    merged = []
    i, j = 0, 0
    
    while i < len(list1) and j < len(list2):
        if list1[i] <= list2[j]:
            merged.append(list1[i])
            i += 1
        else:
            merged.append(list2[j])
            j += 1
    
    # Add remaining elements
    merged.extend(list1[i:])
    merged.extend(list2[j:])
    
    return merged
