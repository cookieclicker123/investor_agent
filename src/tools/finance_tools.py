from datetime import datetime
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import List
from ..data_model import FinanceAgentResponse, StockData, StockPrice, StockFundamentals
from utils.config import get_alpha_vantage_config

def finance_search(query: str) -> FinanceAgentResponse:
    """
    Real finance data from Alpha Vantage API.
    Returns stock data for valid stock symbols.
    """
    try:
        # Initialize session with retry strategy
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
        session.mount('https://', HTTPAdapter(max_retries=retries))

        # API Configuration
        BASE_URL = "https://www.alphavantage.co/query"
        config = get_alpha_vantage_config()
        API_KEY = config["api_key"]

        # Extract symbols (simple implementation matching dummy version)
        words = query.upper().split()
        extracted_symbols = [word.strip(',.!?()') for word in words 
                           if word.strip(',.!?()').isalpha() 
                           and len(word.strip(',.!?()')) <= 5]
        
        # If no symbols found in query, return error
        if not extracted_symbols:
            return FinanceAgentResponse(
                query=query,
                extracted_symbols=[],
                stock_data=[],
                generated_at=datetime.now().isoformat(),
                error="No valid stock symbols found in query. Please provide valid stock symbols."
            )

        # Generate stock data for found symbols
        stock_data: List[StockData] = []
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        for symbol in extracted_symbols:
            try:
                # Get current price data
                quote_params = {
                    "function": "GLOBAL_QUOTE",
                    "symbol": symbol,
                    "apikey": API_KEY
                }
                
                quote_response = session.get(
                    BASE_URL,
                    params=quote_params,
                    timeout=10
                )
                quote_data = quote_response.json()
                
                if "Error Message" in quote_data or "Note" in quote_data:
                    continue
                
                # Get company overview data
                overview_params = {
                    "function": "OVERVIEW",
                    "symbol": symbol,
                    "apikey": API_KEY
                }
                
                overview_response = session.get(
                    BASE_URL,
                    params=overview_params,
                    timeout=10
                )
                overview_data = overview_response.json()
                
                stock_data.append(
                    StockData(
                        symbol=symbol,
                        current_price=StockPrice(
                            price=float(quote_data["Global Quote"]["05. price"]),
                            change_percent=float(quote_data["Global Quote"]["10. change percent"].rstrip('%')),
                            volume=int(quote_data["Global Quote"]["06. volume"]),
                            trading_day=quote_data["Global Quote"]["07. latest trading day"]
                        ),
                        fundamentals=StockFundamentals(
                            market_cap=overview_data.get("MarketCapitalization"),
                            pe_ratio=overview_data.get("PERatio"),
                            eps=overview_data.get("EPS")
                        ),
                        last_updated=datetime.now().isoformat()
                    )
                )
                
            except Exception:
                continue
        
        # If no valid stock data found, return error
        if not stock_data:
            return FinanceAgentResponse(
                query=query,
                extracted_symbols=[],
                stock_data=[],
                generated_at=datetime.now().isoformat(),
                error="No valid stock data found for the provided symbols"
            )

        return FinanceAgentResponse(
            query=query,
            extracted_symbols=[stock.symbol for stock in stock_data],
            stock_data=stock_data,
            generated_at=datetime.now().isoformat()
        )
        
    except Exception as e:
        return FinanceAgentResponse(
            query=query,
            extracted_symbols=[],
            stock_data=[],
            generated_at=datetime.now().isoformat(),
            error=str(e)
        ) 