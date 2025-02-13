import pytest
from datetime import datetime
from src.data_model import WebAgentResponse, SearchResult, LLMResponse, LLMRequest, Intent
from tests.mocks.web_dummy_agent import create_web_agent

@pytest.mark.asyncio
async def test_web_agent_basic_query():
    """Test that web agent returns expected response format"""
    # Arrange
    query = "What are the latest market trends?"
    web_agent = await create_web_agent()
    
    # Act
    response = await web_agent(query)
    
    # Assert
    assert isinstance(response, WebAgentResponse)
    assert any("market" in result.title.lower() or "market" in result.snippet.lower() 
              for result in response.relevant_results)
    assert len(response.search_results) > 0

def test_search_result_model():
    """Test SearchResult model creation and attributes"""
    # Arrange & Act
    result = SearchResult(
        title="Test Title",
        snippet="Test Snippet",
        link="https://test.com",
        date="2024-02-04"
    )
    
    # Assert
    assert isinstance(result, SearchResult)
    assert result.title == "Test Title"
    assert result.snippet == "Test Snippet"
    assert result.link == "https://test.com"
    assert result.date == "2024-02-04"
    
    # Test optional date
    result_no_date = SearchResult(
        title="Test Title",
        snippet="Test Snippet",
        link="https://test.com"
    )
    assert result_no_date.date is None

@pytest.mark.asyncio
async def test_web_agent_integration_with_llm():
    """Test web agent integration with LLMResponse"""
    # Arrange
    query = "What are the latest market trends?"
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
        confidence=0.8
    )
    
    # Assert
    assert llm_response.web_context is not None
    assert isinstance(llm_response.web_context, WebAgentResponse)
    assert len(llm_response.web_context.relevant_results) > 0

@pytest.mark.asyncio
async def test_web_agent_error_handling():
    """Test web agent handles errors gracefully"""
    # Arrange
    async def failing_search_fn(_):
        raise Exception("Search failed")
    
    web_agent = await create_web_agent(failing_search_fn)
    
    # Act & Assert
    with pytest.raises(Exception) as exc_info:
        await web_agent("test query")
    assert "Error in web agent: Search failed" in str(exc_info.value) 