from datetime import datetime
from typing import List
from src.data_model import FinanceAgentResponse, StockData, StockPrice, StockFundamentals

def dummy_finance_search(query: str) -> FinanceAgentResponse:
    """
    Mock finance data for testing and development.
    Returns dummy stock data for common stock symbols.
    """
    # Mock stock database
    MOCK_STOCKS = {
        "AAPL": {
            "price": 150.0,
            "change_percent": 1.5,
            "volume": 1000000,
            "market_cap": "2.5T",
            "pe_ratio": "25.5",
            "eps": "6.5"
        },
        "MSFT": {
            "price": 310.0,
            "change_percent": 0.8,
            "volume": 800000,
            "market_cap": "2.8T",
            "pe_ratio": "30.2",
            "eps": "9.7"
        },
        "GOOGL": {
            "price": 2800.0,
            "change_percent": -0.5,
            "volume": 500000,
            "market_cap": "1.9T",
            "pe_ratio": "28.4",
            "eps": "95.3"
        },
        "TSLA": {
            "price": 180.0,
            "change_percent": 2.3,
            "volume": 1200000,
            "market_cap": "580B",
            "pe_ratio": "45.8",
            "eps": "4.2"
        }
    }

    # Extract symbols from query (simple implementation for dummy data)
    extracted_symbols = [symbol for symbol in MOCK_STOCKS.keys() if symbol in query.upper()]
    
    # If no symbols found in query, return error
    if not extracted_symbols:
        return FinanceAgentResponse(
            query=query,
            extracted_symbols=[],
            stock_data=[],
            generated_at=datetime.now().isoformat(),
            error="No valid stock symbols found in query. Please use symbols like AAPL, MSFT, GOOGL, or TSLA."
        )

    # Generate stock data for found symbols
    stock_data: List[StockData] = []
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    for symbol in extracted_symbols:
        mock_data = MOCK_STOCKS[symbol]
        stock_data.append(
            StockData(
                symbol=symbol,
                current_price=StockPrice(
                    price=mock_data["price"],
                    change_percent=mock_data["change_percent"],
                    volume=mock_data["volume"],
                    trading_day=current_date
                ),
                fundamentals=StockFundamentals(
                    market_cap=mock_data["market_cap"],
                    pe_ratio=mock_data["pe_ratio"],
                    eps=mock_data["eps"]
                ),
                last_updated=datetime.now().isoformat()
            )
        )

    return FinanceAgentResponse(
        query=query,
        extracted_symbols=extracted_symbols,
        stock_data=stock_data,
        generated_at=datetime.now().isoformat()
    )