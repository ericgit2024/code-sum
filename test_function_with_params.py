def format_user_profile(username: str, age: int, email: str, is_premium: bool = False) -> dict:
    """Format user profile data into a standardized dictionary."""
    if not username or len(username.strip()) == 0:
        raise ValueError("Username cannot be empty")
    
    if age < 0 or age > 150:
        raise ValueError("Age must be between 0 and 150")
    
    if '@' not in email:
        raise ValueError("Invalid email format")
    
    # Create profile dictionary
    profile = {
        'username': username.strip().lower(),
        'age': age,
        'email': email.strip().lower(),
        'account_type': 'premium' if is_premium else 'free'
    }
    
    # Add age category
    if age < 18:
        profile['category'] = 'minor'
    elif age < 65:
        profile['category'] = 'adult'
    else:
        profile['category'] = 'senior'
    
    # Premium users get additional fields
    if is_premium:
        profile['features'] = ['ad_free', 'priority_support', 'advanced_analytics']
    else:
        profile['features'] = ['basic_access']
    
    return profile
