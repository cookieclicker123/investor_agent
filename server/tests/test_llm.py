import pytest
import os
from src.data_model import LLMRequest
from tests.mocks.mock_llm import create_mock_llm_client

# Ensure the fixtures directory exists
@pytest.fixture(scope="session", autouse=True)
def setup_fixtures_dir():
    fixtures_dir = os.path.join(os.path.dirname(__file__), "fixtures")
    os.makedirs(fixtures_dir, exist_ok=True)
    return fixtures_dir

@pytest.mark.asyncio
async def test_generate_llm_response_structure():
    """Test that the mock LLM generates responses with correct structure."""
    mock_responses = {
        "What is the current price of (AAPL)?": {  # Changed from "What is AI?"
            "answer": "The current price of AAPL is $150.00",
            "sources": ["Yahoo Finance", "Market Watch"],
            "confidence": 0.95
        }
    }
    
    llm = create_mock_llm_client(query_response=mock_responses, emulation_speed=1000)
    request = LLMRequest(
        query="What is the current price of (AAPL)?",  # Changed from "What is AI?"
        prompt="",
        as_json=True
    )
    
    chunks = []
    def on_chunk(chunk: str):
        chunks.append(chunk)
        assert isinstance(chunk, str)
        assert len(chunk) > 0
    
    response = await llm(request, on_chunk)
    assert response is not None
    assert response.request == request
    assert response.model_provider == "mock"
    assert len(chunks) > 0
    assert isinstance(response.raw_response, dict)
    assert "answer" in response.raw_response
    assert "sources" in response.raw_response
    assert "confidence" in response.raw_response
    assert response.raw_response["confidence"] == 0.95

@pytest.mark.asyncio
async def test_generate_llm_response_chunks():
    """Test that the mock LLM streams responses in chunks."""
    mock_responses = {
        "What is the current price of (AAPL)?": {  # Changed from "What is AI?"
            "answer": "The current price of AAPL is $150.00",
            "sources": ["Yahoo Finance", "Market Watch"],
            "confidence": 0.95
        }
    }
    
    chunks = []
    def on_chunk(chunk: str):
        chunks.append(chunk)
        assert isinstance(chunk, str)
        assert len(chunk) > 0
    
    llm = create_mock_llm_client(query_response=mock_responses, emulation_speed=1000)
    request = LLMRequest(
        query="What is the current price of (AAPL)?",  # Changed from "What is AI?"
        prompt="",
        as_json=True
    )
    
    await llm(request, on_chunk)
    assert len(chunks) > 0
    full_response = "".join(chunks)
    assert isinstance(full_response, str)
    assert len(full_response) > 0

@pytest.mark.asyncio
async def test_conversation_log():
    """Test that the mock LLM properly logs conversations."""
    mock_responses = {
        "What is the current price of (AAPL)?": {  # Changed from "What is AI?"
            "answer": "The current price of AAPL is $150.00",
            "sources": ["Yahoo Finance", "Market Watch"],
            "confidence": 0.95
        }
    }
    
    llm = create_mock_llm_client(query_response=mock_responses, emulation_speed=1000)
    request = LLMRequest(
        query="What is the current price of (AAPL)?",  # Changed from "What is AI?"
        prompt="",
        as_json=True
    )
    
    response = await llm(request, lambda x: None)
    assert response.raw_response == mock_responses[request.query]
