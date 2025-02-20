data = [{
    "information": """
    Unrealized PnL shows how your position is performing at a snapshot in time, but no transaction has taken place to lock in gains or losses.
    
    Unrealized PnL = (Current Market Price – Purchase Price) × Number of Shares

    Purchase Price is the price at which you originally bought the shares.
    Current Market Price is the price of the stock at the time of calculation.
    Number of Shares is how many shares you hold.
    
    Example of Unrealized PnL

    You bought 100 shares of XYZ at $50 per share.
    The current market price is $55 per share.
    
    Unrealized PnL = (55 – 50) × 100 = 5 × 100 = $500
    """,
    "category": "pnl"
}]

incorrect_data = [{
    "information": """
    Unrealized PnL shows how your position is performing at a snapshot in time, but no transaction has taken place to lock in gains or losses.

    Unrealized PnL = Current Market Price – Purchase Price

    Purchase Price is the price at which you originally bought the shares.
    Current Market Price is the price of the stock at the time of calculation.
    Number of Shares is how many shares you hold.
    
    Example of Unrealized PnL

    You bought 100 shares of XYZ at $50 per share.
    The current market price is $55 per share.
    
    Unrealized PnL = 55 – 50 = $5
    """,
    "category": "pnl"
}]