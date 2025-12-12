def process_user_data(user_id: int, 
                     data: dict, 
                     options: dict = None,
                     validate: bool = True,
                     transform: bool = True,
                     save_to_db: bool = True) -> dict:
    # Initialize result dictionary
    result = {
        'user_id': user_id,
        'status': 'pending',
        'errors': [],
        'warnings': [],
        'processed_data': None
    }
    
    # Set default options
    if options is None:
        options = {
            'strict_mode': False,
            'max_retries': 3,
            'timeout': 30,
            'cache_enabled': True
        }
    
    # Validation phase
    if validate:
        # Check user_id
        if not isinstance(user_id, int) or user_id <= 0:
            result['errors'].append('Invalid user_id: must be positive integer')
            result['status'] = 'failed'
            return result
        
        # Check data structure
        required_fields = ['name', 'email', 'age']
        missing_fields = [f for f in required_fields if f not in data]
        
        if missing_fields:
            result['errors'].append(f'Missing required fields: {missing_fields}')
            result['status'] = 'failed'
            return result
        
        # Validate email format
        email = data.get('email', '')
        if '@' not in email or '.' not in email:
            result['errors'].append('Invalid email format')
            if options.get('strict_mode'):
                result['status'] = 'failed'
                return result
            else:
                result['warnings'].append('Email format questionable')
        
        # Validate age range
        age = data.get('age')
        if not isinstance(age, int) or age < 0 or age > 150:
            result['errors'].append('Invalid age: must be between 0 and 150')
            result['status'] = 'failed'
            return result
    
    # Transformation phase
    if transform:
        processed = {}
        
        # Normalize name
        name = data.get('name', '').strip()
        processed['name'] = ' '.join(name.split()).title()
        
        # Normalize email
        email = data.get('email', '').strip().lower()
        processed['email'] = email
        
        # Calculate age category
        age = data.get('age', 0)
        if age < 18:
            processed['age_category'] = 'minor'
        elif age < 65:
            processed['age_category'] = 'adult'
        else:
            processed['age_category'] = 'senior'
        
        processed['age'] = age
        
        # Add metadata
        import datetime
        processed['processed_at'] = datetime.datetime.now().isoformat()
        processed['processor_version'] = '2.1.0'
        
        result['processed_data'] = processed
    else:
        result['processed_data'] = data.copy()
    
    # Database persistence phase
    if save_to_db:
        try:
            # Simulate database connection
            max_retries = options.get('max_retries', 3)
            timeout = options.get('timeout', 30)
            
            for attempt in range(max_retries):
                try:
                    # Simulate DB save operation
                    cache_key = f"user_{user_id}_data"
                    
                    if options.get('cache_enabled'):
                        # Simulate cache update
                        result['warnings'].append('Cache updated')
                    
                    # Success
                    result['status'] = 'success'
                    result['db_saved'] = True
                    break
                    
                except Exception as e:
                    if attempt == max_retries - 1:
                        result['errors'].append(f'DB save failed after {max_retries} attempts')
                        result['status'] = 'partial'
                    else:
                        # Retry
                        import time
                        time.sleep(0.1 * (attempt + 1))
                        
        except Exception as e:
            result['errors'].append(f'Database error: {str(e)}')
            result['status'] = 'partial'
    else:
        result['status'] = 'success'
        result['db_saved'] = False
    
    # Final validation
    if result['errors']:
        result['status'] = 'failed' if not result.get('processed_data') else 'partial'
    
    return result
