from datetime import datetime
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import List
from ..data_model import FinanceAgentResponse, StockData, StockPrice, StockFundamentals
from utils.config import get_alpha_vantage_config
import re

def extract_stock_symbols(query: str) -> List[str]:
    """Extract stock symbols with strict formatting requirements"""
    symbols = set()
    
    # Primary check: Look for properly formatted symbols in parentheses
    parens_pattern = r'\(([A-Z]{1,4})\)'  # 1-4 capital letters in parentheses
    parens_symbols = set(re.findall(parens_pattern, query))
    if parens_symbols:
        symbols.update(parens_symbols)
        return list(symbols)  # Return early if we find properly formatted symbols
        
    # Secondary check: Look for standalone capital letters that match stock pattern
    standalone_pattern = r'\b[A-Z]{1,4}\b'  # 1-4 capital letters as whole word
    standalone_matches = set(re.findall(standalone_pattern, query))
    
    # Filter standalone matches through additional validation
    for match in standalone_matches:
        if looks_like_stock_symbol(match):
            symbols.add(match)
    
    if not symbols:
        raise ValueError(
            "No valid stock symbols found. Please use format (AAPL) or AAPL. "
            "Examples: (MSFT), (GOOGL), AAPL, TSLA"
        )
    
    return list(symbols)

def looks_like_stock_symbol(text: str) -> bool:
    """Stricter validation for potential stock symbols"""
    if not (1 <= len(text) <= 4 and text.isalpha() and text.isupper()):
        return False
        
    # Additional validation checks
    if text.endswith('S'):  # Plural words
        return False
    if len(text) == 1:  # Single letters
        return False
    if all(c == text[0] for c in text):  # Repeated letters
        return False
        
    return True

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

        # Extract symbols using robust method
        try:
            extracted_symbols = extract_stock_symbols(query)
        except ValueError as e:
            return FinanceAgentResponse(
                query=query,
                extracted_symbols=[],
                stock_data=[],
                generated_at=datetime.now().isoformat(),
                error=str(e)
            )

        # Generate stock data for valid symbols
        stock_data: List[StockData] = []
        
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

        if not stock_data:
            return FinanceAgentResponse(
                query=query,
                extracted_symbols=extracted_symbols,
                stock_data=[],
                generated_at=datetime.now().isoformat(),
                error="No valid stock data found for the provided symbols"
            )

        return FinanceAgentResponse(
            query=query,
            extracted_symbols=extracted_symbols,
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