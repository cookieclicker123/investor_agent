import pytest
from unittest.mock import patch
import logging
from server.src.data_model import LLMRequest, Intent
from server.src.ollama_llm import create_ollama_llm

logger = logging.getLogger(__name__)

@pytest.mark.asyncio
async def test_ollama_basic():
    """Test basic Ollama LLM functionality."""
    llm = create_ollama_llm()
    
    request = LLMRequest(
        query="test query",
        prompt="test prompt"
    )
    
    response = await llm(request, lambda x: None)
    assert response is not None
    assert response.model_provider == "ollama"

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
    # Check if query appears in meta prompt content
    assert request.query in response.request.prompt["meta_agent"]["raw_text"]

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
        assert "error" in str(response.raw_response["raw_text"])
        assert "test error" in str(response.raw_response["raw_text"]).lower()
        assert response.intents == [Intent.WEB_AGENT]  # Default intent
        assert response.confidence == 0.0  # Error case should have zero confidence

@pytest.mark.asyncio
async def test_ollama_intent_handling():
    """Test handling of different intents."""
    llm = create_ollama_llm()
    
    test_cases = [
        ("What is the current price of (AAPL)?", [Intent.FINANCE_AGENT]),
        ("How do options work?", [Intent.PDF_AGENT, Intent.WEB_AGENT]),
        ("What's happening in the market today?", [Intent.WEB_AGENT])
    ]
    
    for query, expected_intents in test_cases:
        request = LLMRequest(
            query=query,
            prompt="",
            as_json=True
        )
        
        response = await llm(request, lambda x: None)
        logger.debug(f"Response raw_text: {response.raw_response.get('raw_text', '')}")
        logger.debug(f"Detected intents: {response.intents}")
        # Check if any of the expected intents are present
        assert any(intent in response.intents for intent in expected_intents)

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
    assert len(chunks) > 0  # Verify we got chunks
    assert response.intents  # Verify we got some intents
    assert response.confidence > 0  # Verify we got a confidence score
    # Check if finance intent is present for stock query
    assert Intent.FINANCE_AGENT in response.intents 