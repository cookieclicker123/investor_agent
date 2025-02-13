import pytest
import logging
from server.src.data_model import LLMRequest, Intent
from server.src.ollama_llm import create_ollama_llm
from server.src.groq_llm import create_groq_llm

logger = logging.getLogger(__name__)

@pytest.mark.asyncio
class TestLLMAgents:
    """Test suite for LLM agent functionality."""
    
    @pytest.fixture
    def ollama_llm(self):
        return create_ollama_llm()
        
    @pytest.fixture
    def groq_llm(self):
        return create_groq_llm()
    
    async def test_ollama_basic_finance_intent(self, ollama_llm):
        """Test basic finance query with stock symbol using Ollama."""
        
        logger.debug("Starting basic finance intent test")
        
        chunks = []
        def on_chunk(chunk: str):
            chunks.append(chunk)
            logger.debug(f"Received chunk: {chunk}")
        
        request = LLMRequest(
            query="What is the current price of (AAPL)?",
            prompt="",
            as_json=True
        )
        logger.debug(f"Created test request: {request}")
        
        response = await ollama_llm(request, on_chunk)
        logger.debug(f"Final response: {response}")
        logger.debug(f"Detected intents: {response.intents}")
        
        # Basic assertions
        assert response is not None
        assert response.raw_response is not None
        assert response.intents is not None
        assert len(response.intents) > 0
        assert Intent.FINANCE_AGENT in response.intents
        
        # Verify prompt structure
        assert isinstance(response.request.prompt, dict)
        assert "meta_agent" in response.request.prompt
        assert "selected_agent" in response.request.prompt
        
        # Verify streaming worked
        assert len(chunks) > 0
        
    async def test_groq_basic_finance_intent(self, groq_llm):
        """Test basic finance query with stock symbol using Groq."""
        
        chunks = []
        def on_chunk(chunk: str):
            chunks.append(chunk)
            logger.debug(f"Received Groq chunk: {chunk}")
        
        request = LLMRequest(
            query="What is the current price of (AAPL)?",
            prompt="",
            as_json=True
        )
        
        response = await groq_llm(request, on_chunk)
        
        logger.debug(f"Groq meta agent response: {response.raw_response}")
        logger.debug(f"Groq detected intents: {response.intents}")
        
        # Basic assertions
        assert response is not None
        assert response.raw_response is not None
        assert response.intents is not None
        assert len(response.intents) > 0
        assert Intent.FINANCE_AGENT in response.intents
        
        # Verify prompt structure
        assert isinstance(response.request.prompt, dict)
        assert "meta_agent" in response.request.prompt
        assert "selected_agent" in response.request.prompt
        
        # Verify streaming worked
        assert len(chunks) > 0