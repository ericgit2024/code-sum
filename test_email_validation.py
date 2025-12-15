def validate_email(email: str) -> bool:
    """Validate an email address."""
    if not email or '@' not in email:
        return False
    
    parts = email.split('@')
    if len(parts) != 2:
        return False
    
    username, domain = parts
    
    if not username or not domain:
        return False
    
    if '.' not in domain:
        return False
    
    return True
