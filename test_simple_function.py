def calculate_discount(price: float, discount_percent: int, is_member: bool = False) -> float:
    if price < 0 or discount_percent < 0 or discount_percent > 100:
        raise ValueError("Invalid price or discount percentage")
    
    discount_amount = price * (discount_percent / 100)
    final_price = price - discount_amount
    
    if is_member:
        final_price = final_price * 0.95
    
    return round(final_price, 2)
