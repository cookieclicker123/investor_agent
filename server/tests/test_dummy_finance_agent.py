import pytest
from datetime import datetime
from server.src.data_model import (
    FinanceAgentResponse, StockData, StockPrice, StockFundamentals,
    LLMResponse, LLMRequest, Intent
)
from server.tests.mocks.finance_dummy_agent import create_finance_agent

def test_finance_agent_basic_query():
    """Test that finance agent returns expected response format"""
    # Arrange
    query = "What's the price of AAPL?"
    finance_agent = create_finance_agent()
    
    # Act
    response = finance_agent(query)
    
    # Assert
    assert isinstance(response, FinanceAgentResponse)
    assert "AAPL" in response.extracted_symbols
    assert len(response.stock_data) > 0
    assert response.generated_at is not None
    assert response.error is None

def test_stock_data_models():
    """Test stock-related model creation and attributes"""
    # Arrange & Act
    price = StockPrice(
        price=150.0,
        change_percent=1.5,
        volume=1000000,
        trading_day="2024-02-04"
    )
    
    fundamentals = StockFundamentals(
        market_cap="2.5T",
        pe_ratio="25.5",
        eps="6.5"
    )
    
    stock_data = StockData(
        symbol="AAPL",
        current_price=price,
        fundamentals=fundamentals,
        last_updated=datetime.now().isoformat()
    )
    
    # Assert
    assert isinstance(stock_data, StockData)
    assert stock_data.symbol == "AAPL"
    assert stock_data.current_price.price == 150.0
    assert stock_data.fundamentals.market_cap == "2.5T"
    
    # Test optional fields
    fundamentals_optional = StockFundamentals(
        market_cap=None,
        pe_ratio=None,
        eps=None
    )
    assert all(getattr(fundamentals_optional, field) is None 
              for field in ["market_cap", "pe_ratio", "eps"])

def test_finance_agent_integration_with_llm():
    """Test finance agent integration with LLMResponse"""
    # Arrange
    query = "Compare AAPL and MSFT"
    llm_request = LLMRequest(
        query=query,
        prompt="Test prompt",
        as_json=True
    )
    
    # Create LLMResponse with finance context
    finance_agent = create_finance_agent()
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
    assert "AAPL" in llm_response.finance_context.extracted_symbols
    assert "MSFT" in llm_response.finance_context.extracted_symbols

def test_finance_agent_error_handling():
    """Test finance agent handles errors gracefully"""
    # Arrange
    def failing_search_fn(_):
        raise Exception("Finance search failed")
    
    finance_agent = create_finance_agent(use_dummy=False)  # This will trigger error as no real search is configured
    
    # Act
    response = finance_agent("test query")
    
    # Assert
    assert isinstance(response, FinanceAgentResponse)
    assert response.error is not None
    assert "No finance search function configured" in response.error
    assert len(response.stock_data) == 0
    assert len(response.extracted_symbols) == 0

def test_finance_agent_invalid_symbols():
    """Test finance agent handles invalid stock symbols"""
    # Arrange
    query = "What's the price of INVALID?"
    finance_agent = create_finance_agent()
    
    # Act
    response = finance_agent(query)
    
    # Assert
    assert isinstance(response, FinanceAgentResponse)
    assert response.error is not None
    assert len(response.stock_data) == 0
    assert len(response.extracted_symbols) == 0

def test_finance_agent_multiple_symbols():
    """Test finance agent handles multiple stock symbols"""
    # Arrange
    query = "Compare AAPL, MSFT, and GOOGL"
    finance_agent = create_finance_agent()
    
    # Act
    response = finance_agent(query)
    
    # Assert
    assert isinstance(response, FinanceAgentResponse)
    assert len(response.extracted_symbols) == 3
    assert all(symbol in response.extracted_symbols for symbol in ["AAPL", "MSFT", "GOOGL"])
    assert len(response.stock_data) == 3
    assert all(isinstance(data, StockData) for data in response.stock_data)
