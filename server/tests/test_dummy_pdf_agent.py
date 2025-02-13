import pytest
from src.groq_llm import create_groq_llm
from src.ollama_llm import create_ollama_llm
from tests.mocks.pdf_dummy_agent import PDFDummyAgent
from src.data_model import PDFContext, PDFAgentResponse, LLMRequest, LLMResponse
from src.data_model import Intent

@pytest.mark.asyncio
async def test_llm_with_pdf_context():
    """Test LLMs can handle PDF context in responses"""
    query = "What are the best options trading strategies?"
    
    # Get mock PDF context
    pdf_agent = PDFDummyAgent()
    context = pdf_agent.process_query(query)
    
    # Create LLMRequest
    request = LLMRequest(
        prompt="",  # Will be filled by LLM
        query=query,
        as_json=True
    )
    
    # Test Groq LLM implementation only
    llm = create_groq_llm()
    response = await llm(llm_request=request)
    
    # Updated assertions to match new model
    assert response.intents is not None
    assert len(response.intents) > 0
    assert response.raw_response != ""
    assert response.confidence > 0.0
    # Removed model assertion as it's not in our data model

def test_pdf_context_structure():
    """Test PDF context structure and formatting"""
    pdf_agent = PDFDummyAgent()
    context = pdf_agent.process_query("test query")
    
    # Test context structure
    assert isinstance(context, PDFAgentResponse)
    assert len(context.relevant_chunks) > 0
    for chunk in context.relevant_chunks:
        assert isinstance(chunk, PDFContext)
        assert 0 <= chunk.similarity_score <= 1
        assert chunk.chunk_id >= 0
        assert chunk.total_chunks > 0

@pytest.mark.asyncio
async def test_llm_response_integration():
    """Test LLM response with different context scenarios"""
    query = "How do options trading work?"
    pdf_agent = PDFDummyAgent()
    context = pdf_agent.process_query(query)
    
    llm = create_groq_llm()
    
    # Test with PDF intent
    request_with_pdf = LLMRequest(
        prompt="",
        query=query,
        pdf_context=context
    )
    response = await llm(llm_request=request_with_pdf)
    assert response.intents is not None
    assert response.confidence > 0.0
    
    # Test without PDF intent
    request_without_pdf = LLMRequest(
        prompt="",
        query=query
    )
    response = await llm(llm_request=request_without_pdf)
    # PDF context should match intent detection
    if Intent.PDF_AGENT in response.intents:
        assert response.pdf_context is not None
    else:
        assert response.pdf_context is None

def test_context_formatting():
    """Test context formatting for LLM consumption"""
    pdf_agent = PDFDummyAgent()
    context = pdf_agent.process_query("test query")
    
    formatted = pdf_agent.format_for_llm(context)
    assert isinstance(formatted, str)
    assert "Source:" in formatted
    assert "Relevance:" in formatted
    assert "Content:" in formatted 

@pytest.mark.asyncio
async def test_llm_response_object_integration():
    """Test full LLMResponse object integration with PDF context"""
    query = "How do options trading work?"
    pdf_agent = PDFDummyAgent()
    context = pdf_agent.process_query(query)
    
    llm = create_groq_llm()
    request = LLMRequest(
        prompt="",
        query=query,
        pdf_context=context
    )
    response = await llm(llm_request=request)
    
    # Verify full LLMResponse structure
    assert isinstance(response, LLMResponse)
    assert response.request.query == query
    assert isinstance(response.raw_response, dict)
    assert response.model_name in ["deepseek-r1-distill-llama-70b"]
    assert response.intents is not None
    assert response.confidence > 0.0
    
    # Verify PDF context integration
    assert response.pdf_context is not None
    assert isinstance(response.pdf_context, PDFAgentResponse)
    assert len(response.pdf_context.relevant_chunks) > 0
    
    # Verify chunk details
    for chunk in response.pdf_context.relevant_chunks:
        assert isinstance(chunk, PDFContext)
        assert chunk.text
        assert chunk.source_file
        assert 0 <= chunk.similarity_score <= 1
        assert chunk.chunk_id >= 0
        assert chunk.total_chunks > 0 