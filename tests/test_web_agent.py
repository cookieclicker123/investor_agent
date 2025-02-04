import pytest
import asyncio
from datetime import datetime
from src.data_model import (
    WebAgentResponse, SearchResult,
    LLMResponse, LLMRequest, Intent
)
from src.agents.web_agent import create_web_agent
from src.tools.web_tools import create_web_search

@pytest.mark.asyncio
async def test_real_web_agent_basic_query():
    """Test that real web agent returns expected response format"""
    # Arrange
    query = "What are the latest stock market trends?"
    web_agent = await create_web_agent()  # Use real search by default
    
    # Act
    response = await web_agent(query)
    
    # Assert
    assert isinstance(response, WebAgentResponse)
    assert len(response.search_results) > 0
    assert response.generated_at is not None
    assert response.error is None
    
    # Check real data structure
    result = response.search_results[0]
    assert isinstance(result, SearchResult)
    assert result.title is not None
    assert result.snippet is not None
    assert result.link is not None
    assert result.date is not None

@pytest.mark.asyncio
async def test_real_web_agent_integration_with_llm():
    """Test real web agent integration with LLMResponse"""
    # Arrange
    query = "What are current interest rates?"
    llm_request = LLMRequest(
        query=query,
        prompt="Test prompt",
        as_json=True
    )
    
    # Create LLMResponse with web context
    web_agent = await create_web_agent()
    web_context = await web_agent(query)
    
    llm_response = LLMResponse(
        generated_at=datetime.now().isoformat(),
        intents=[Intent.WEB_AGENT],
        request=llm_request,
        raw_response={"text": "Test response"},
        model_name="test-model",
        model_provider="test-provider",
        time_in_seconds=0.1,
        pdf_context=None,
        web_context=web_context,
        finance_context=None,
        confidence=0.8
    )
    
    # Assert
    assert llm_response.web_context is not None
    assert isinstance(llm_response.web_context, WebAgentResponse)
    assert len(llm_response.web_context.search_results) > 0

@pytest.mark.asyncio
async def test_real_web_agent_empty_query():
    """Test web agent handles empty queries"""
    # Arrange
    query = ""
    web_agent = await create_web_agent()
    
    # Act
    response = await web_agent(query)
    
    # Assert
    assert isinstance(response, WebAgentResponse)
    assert response.error is not None
    assert len(response.search_results) == 0

@pytest.mark.asyncio
async def test_real_web_agent_rate_limiting():
    """Test web agent handles rate limiting"""
    # Arrange
    queries = [
        "Stock market news",
        "Interest rates",
        "Federal Reserve",
        "Market trends",
        "Economic outlook"
    ]
    web_agent = await create_web_agent()
    
    # Act & Assert
    for query in queries:
        response = await web_agent(query)
        assert isinstance(response, WebAgentResponse)
        if response.error:
            assert "rate limit" in response.error.lower()
            break
        else:
            assert len(response.search_results) > 0

@pytest.mark.asyncio
async def test_real_web_agent_result_filtering():
    """Test web agent properly filters and formats results"""
    # Arrange
    query = "Latest financial news from reputable sources"
    web_agent = await create_web_agent()
    
    # Act
    response = await web_agent(query)
    
    # Assert
    assert isinstance(response, WebAgentResponse)
    for result in response.search_results:
        # Check date formatting
        assert isinstance(result.date, str)
        # Verify URLs are valid
        assert result.link.startswith(('http://', 'https://'))
        # Check content isn't empty
        assert len(result.snippet) > 0
        assert len(result.title) > 0

@pytest.mark.asyncio
async def test_real_web_agent_error_handling():
    """Test web agent handles various error conditions"""
    # Arrange
    web_agent = await create_web_agent()
    
    # Test with very long query
    long_query = "x" * 1000
    response = await web_agent(long_query)
    assert response.error is not None
    
    # Test with special characters
    special_query = "!@#$%^&*()"
    response = await web_agent(special_query)
    assert len(response.search_results) == 0
    
    # Test with non-string input
    with pytest.raises(TypeError):
        await web_agent(123) 