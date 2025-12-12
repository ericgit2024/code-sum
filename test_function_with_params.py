def process_payment(amount: float, currency: str = "USD", apply_tax: bool = True) -> dict:
    """Process a payment transaction with optional tax calculation."""
    if amount <= 0:
        raise ValueError("Amount must be positive")
    
    if currency not in ["USD", "EUR", "GBP"]:
        raise ValueError("Unsupported currency")
    
    # Calculate tax if applicable
    if apply_tax:
        tax_rate = 0.10  # 10% tax
        tax_amount = amount * tax_rate
        total = amount + tax_amount
    else:
        tax_amount = 0.0
        total = amount
    
    # Create transaction record
    transaction = {
        "amount": round(amount, 2),
        "currency": currency,
        "tax": round(tax_amount, 2),
        "total": round(total, 2),
        "taxed": apply_tax
    }
    
    return transaction
