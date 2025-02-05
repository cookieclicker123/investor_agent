"""
DO NOT RUN THIS TEST TOO MANY API CALLS FIXING IT. 15 only get 25 calls a day on finance, the dummy worked fine so all good.
"""

import pytest
import os
from datetime import datetime
from src.data_model import (
    FinanceAgentResponse, StockData,
    LLMResponse, LLMRequest, Intent
)
from src.agents.finance_agent import create_finance_agent

@pytest.fixture(autouse=True)
def check_api_key():
    """Ensure API key is available and valid before running tests"""
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        pytest.skip("ALPHA_VANTAGE_API_KEY not found in environment")
    print(f"Using API key: {api_key[:4]}...{api_key[-4:]}")  # Debug log first/last 4 chars

def test_api_connectivity():
    """Test basic API connectivity before running other tests"""
    # Arrange
    query = "AAPL"  # Use a reliable stock
    finance_agent = create_finance_agent(use_dummy=False)
    
    # Act
    response = finance_agent(query)
    
    # Assert & Debug
    print(f"API Response: {response}")  # Debug full response
    if response.error:
        print(f"API Error: {response.error}")
    assert response.error is None, f"API call failed: {response.error}"
    assert len(response.stock_data) > 0

def test_real_finance_agent_basic_query():
    """Test that real finance agent returns expected response format"""
    # Arrange
    query = "What's the price of AAPL?"
    finance_agent = create_finance_agent(use_dummy=False)
    
    # Act
    response = finance_agent(query)
    
    # Debug logging
    print(f"Extracted symbols: {response.extracted_symbols}")
    print(f"Stock data length: {len(response.stock_data)}")
    if response.error:
        print(f"Error: {response.error}")
    
    # Assert
    assert isinstance(response, FinanceAgentResponse)
    assert "AAPL" in response.extracted_symbols, "Symbol extraction failed"
    assert len(response.stock_data) > 0, "No stock data returned"
    assert response.generated_at is not None
    assert response.error is None, f"Unexpected error: {response.error}"
    
    # Check real data structure
    stock = response.stock_data[0]
    assert isinstance(stock.current_price.price, float)
    assert isinstance(stock.current_price.volume, int)
    assert isinstance(stock.current_price.change_percent, float)

def test_real_stock_data_models():
    """Test real stock data contains valid information"""
    # Arrange
    query = "Compare MSFT stock"
    finance_agent = create_finance_agent(use_dummy=False)
    
    # Act
    response = finance_agent(query)
    stock_data = response.stock_data[0]
    
    # Assert
    assert isinstance(stock_data, StockData)
    assert stock_data.symbol == "MSFT"
    assert stock_data.current_price.price > 0
    assert isinstance(stock_data.fundamentals.market_cap, str)
    assert isinstance(stock_data.fundamentals.pe_ratio, str)
    assert stock_data.last_updated is not None

def test_real_finance_agent_integration_with_llm():
    """Test real finance agent integration with LLMResponse"""
    # Arrange
    query = "Compare AAPL and MSFT"
    llm_request = LLMRequest(
        query=query,
        prompt="Test prompt",
        as_json=True
    )
    
    # Create LLMResponse with finance context
    finance_agent = create_finance_agent(use_dummy=False)
    finance_context = finance_agent(query)
    
    llm_response = LLMResponse(
        generated_at=datetime.now().isoformat(),
        intents=[Intent.FINANCE_AGENT],
        request=llm_request,
        raw_response={"text": "Test response"},
        model_name="test-model",
        model_provider="test-provider",
        time_in_seconds=0.1,
        pdf_context=None,
        web_context=None,
        finance_context=finance_context,
        confidence=0.8
    )
    
    # Assert
    assert llm_response.finance_context is not None
    assert isinstance(llm_response.finance_context, FinanceAgentResponse)
    assert len(llm_response.finance_context.extracted_symbols) == 2
    assert all(symbol in ["AAPL", "MSFT"] for symbol in llm_response.finance_context.extracted_symbols)

def test_real_finance_agent_invalid_symbols():
    """Test real finance agent handles invalid stock symbols"""
    # Arrange
    query = "What's the price of INVALID?"
    finance_agent = create_finance_agent(use_dummy=False)
    
    # Act
    response = finance_agent(query)
    
    # Assert
    assert isinstance(response, FinanceAgentResponse)
    assert response.error is not None
    assert len(response.stock_data) == 0
    assert len(response.extracted_symbols) == 0

def test_real_finance_agent_api_rate_limit():
    """Test finance agent handles API rate limiting"""
    # Arrange
    queries = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]  # Multiple queries to potentially trigger rate limit
    finance_agent = create_finance_agent(use_dummy=False)
    
    # Act & Assert
    for query in queries:
        response = finance_agent(query)
        assert isinstance(response, FinanceAgentResponse)
        if response.error:
            assert "API rate limit" in response.error
            break
        else:
            assert len(response.stock_data) > 0

def test_real_finance_agent_multiple_symbols():
    """Test real finance agent handles multiple valid stock symbols"""
    # Arrange
    query = "Compare AAPL, MSFT, and GOOGL"
    finance_agent = create_finance_agent(use_dummy=False)
    
    # Act
    response = finance_agent(query)
    
    # Assert
    assert isinstance(response, FinanceAgentResponse)
    assert len(response.extracted_symbols) > 0
    assert all(isinstance(data, StockData) for data in response.stock_data)
    
    # Check each stock has valid data
    for stock in response.stock_data:
        assert stock.current_price.price > 0
        assert stock.current_price.volume > 0
        assert isinstance(stock.fundamentals.market_cap, str)

def test_api_error_handling():
    """Test specific API error scenarios"""
    # Test with invalid API key
    original_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    os.environ["ALPHA_VANTAGE_API_KEY"] = "invalid_key"
    
    finance_agent = create_finance_agent(use_dummy=False)
    response = finance_agent("AAPL")
    
    print(f"Invalid key response: {response}")  # Debug log
    assert response.error is not None
    assert "API" in response.error.lower()
    
    # Restore original key
    os.environ["ALPHA_VANTAGE_API_KEY"] = original_key 



