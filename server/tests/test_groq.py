import pytest
from src.data_model import LLMRequest, Intent
from src.groq_llm import create_groq_llm
from src.intent_extraction import create_intent_detector

@pytest.mark.asyncio
async def test_groq_basic_response():
    """Test basic Groq response generation."""
    llm = create_groq_llm()
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
    assert response.model_provider == "groq"
    assert response.model_name == "deepseek-r1-distill-llama-70b"
    assert Intent.FINANCE_AGENT in response.intents
    assert isinstance(response.raw_response, dict)

@pytest.mark.asyncio
async def test_groq_prompt_formatting():
    """Test that prompts are properly formatted and included."""
    llm = create_groq_llm()
    
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
    assert Intent.PDF_AGENT in response.intents

@pytest.mark.asyncio
async def test_groq_intent_handling():
    """Test handling of different intents."""
    llm = create_groq_llm()
    
    test_cases = [
        ("What is the current price of (AAPL)?", Intent.FINANCE_AGENT),
        ("How do options trading work?", Intent.PDF_AGENT),
        ("What's happening in the market today?", Intent.WEB_AGENT)
    ]
    
    for query, expected_intent in test_cases:
        request = LLMRequest(
            query=query,
            prompt="",
            as_json=True
        )
        
        response = await llm(request, lambda x: None)
        assert expected_intent in response.intents
        assert response.request.prompt["selected_agent"] is not None

@pytest.mark.asyncio
async def test_groq_streaming():
    """Test that streaming works correctly."""
    llm = create_groq_llm()
    
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
    assert Intent.FINANCE_AGENT in response.intents
    assert response.request.prompt["selected_agent"] is not None
    assert len(chunks) > 0
