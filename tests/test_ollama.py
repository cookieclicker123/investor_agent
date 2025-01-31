import pytest
from src.data_model import LLMRequest, Intent
from src.ollama_llm import create_ollama_llm
from src.intent_extraction import create_intent_detector
from unittest.mock import patch
import logging

logger = logging.getLogger(__name__)

@pytest.mark.asyncio
async def test_ollama_basic_response():
    """Test basic Ollama response generation."""
    llm = create_ollama_llm()
    detect_intent = create_intent_detector()
    
    chunks = []
    def on_chunk(chunk: str):
        chunks.append(chunk)
        
    query = "What is the current price of (AAPL)?"
    intent_result = detect_intent(query)
    
    request = LLMRequest(
        query=query,
        prompt="",  # Will be filled by LLM
        as_json=True
    )
    
    response = await llm(request, on_chunk)
    
    assert response is not None
    assert response.model_provider == "ollama"
    assert response.model_name == "llama3.2:3b"
    assert Intent.FINANCE_AGENT in response.intent  # Check detected intent
    assert isinstance(response.raw_response, dict)

@pytest.mark.asyncio
async def test_ollama_prompt_formatting():
    """Test that prompts are properly formatted and included."""
    llm = create_ollama_llm()
    
    request = LLMRequest(
        query="How do options trading work?",
        prompt="",
        as_json=True
    )
    
    response = await llm(request, lambda x: None)
    
    assert isinstance(response.request.prompt, dict)
    assert "meta_agent" in response.request.prompt
    assert "selected_agent" in response.request.prompt
    assert request.query in response.request.prompt["meta_agent"]["raw_text"]
    assert Intent.PDF_AGENT in response.intent  # Educational query should trigger PDF agent

@pytest.mark.asyncio
async def test_ollama_error_handling():
    """Test that errors are properly handled and formatted."""
    llm = create_ollama_llm()
    
    request = LLMRequest(
        query="test query",
        prompt="",
        as_json=True
    )
    
    with patch('src.llms.ollama._stream_ollama_response') as mock_request:
        mock_request.side_effect = Exception("Test error")
        response = await llm(request, lambda x: None)
        
        assert isinstance(response.raw_response, dict)
        assert "raw_text" in response.raw_response
        assert "error" in response.raw_response["raw_text"].lower()
        assert "test error" in response.raw_response["raw_text"].lower()
        
        assert response.intent == []
        assert response.time_in_seconds >= 0
        assert response.model_provider == "ollama"

@pytest.mark.asyncio
async def test_ollama_intent_handling():
    """Test handling of different intents."""
    llm = create_ollama_llm()
    detect_intent = create_intent_detector()
    
    test_cases = [
        ("What is the current price of (AAPL)?", Intent.FINANCE_AGENT),
        ("How do options work?", Intent.PDF_AGENT),
        ("What's happening in the market today?", Intent.WEB_AGENT)
    ]
    
    for query, expected_intent in test_cases:
        request = LLMRequest(
            query=query,
            prompt="",
            as_json=True
        )
        
        response = await llm(request, lambda x: None)
        logger.debug(f"Response raw_text: {response.raw_response.get('raw_text', '')}")
        logger.debug(f"Detected intent: {response.intent}")
        assert expected_intent in response.intent
        assert response.request.prompt["selected_agent"] is not None

@pytest.mark.asyncio
async def test_ollama_streaming():
    """Test that streaming works correctly."""
    llm = create_ollama_llm()
    
    chunks = []
    def on_chunk(chunk: str):
        chunks.append(chunk)
        assert isinstance(chunk, str)
        assert len(chunk) > 0
    
    request = LLMRequest(
        query="What is the current price of (AAPL)?",
        prompt="",
        as_json=True
    )
    
    response = await llm(request, on_chunk)
    assert Intent.FINANCE_AGENT in response.intent  # Verify correct intent detected
    assert response.request.prompt["selected_agent"] is not None  # Verify prompt was selected 