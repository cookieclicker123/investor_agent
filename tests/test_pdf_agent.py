import pytest
from src.agents.pdf_agent import create_pdf_agent
from src.data_model import PDFAgentResponse, PDFContext

@pytest.mark.asyncio
async def test_pdf_agent_response_shape():
    """Test that PDF agent returns correct response shape"""
    pdf_agent = create_pdf_agent()
    
    response = await pdf_agent("How do options trading work?")
    
    # Check response type
    assert isinstance(response, PDFAgentResponse)
    
    # Check response structure
    assert hasattr(response, 'relevant_chunks')
    assert hasattr(response, 'synthesized_answer')
    
    # Check chunks are properly structured
    for chunk in response.relevant_chunks:
        assert isinstance(chunk, PDFContext)
        assert hasattr(chunk, 'text')
        assert hasattr(chunk, 'source_file')
        assert hasattr(chunk, 'chunk_id')
        assert hasattr(chunk, 'total_chunks')
        assert hasattr(chunk, 'similarity_score')
        
        # Check data types
        assert isinstance(chunk.text, str)
        assert isinstance(chunk.source_file, str)
        assert isinstance(chunk.chunk_id, int)
        assert isinstance(chunk.total_chunks, int)
        assert isinstance(chunk.similarity_score, float)
        
        # Check score range
        assert 0 <= chunk.similarity_score <= 1

@pytest.mark.asyncio
async def test_pdf_agent_prompt_formatting():
    """Test that synthesized answer contains properly formatted prompt"""
    pdf_agent = create_pdf_agent()
    
    response = await pdf_agent("How do options trading work?")
    
    assert response.synthesized_answer is not None
    assert "Context Documents:" in response.synthesized_answer
    assert "Query:" in response.synthesized_answer
    assert "Source:" in response.synthesized_answer
    assert "Relevance:" in response.synthesized_answer

@pytest.mark.asyncio
async def test_pdf_agent_error_handling():
    """Test that PDF agent handles errors gracefully"""
    pdf_agent = create_pdf_agent()
    
    # Test with empty query
    response = await pdf_agent("")
    assert response is None
    
    # Test with very long query
    long_query = "what is " * 1000
    response = await pdf_agent(long_query)
    assert response is not None  # Should still work with long queries

@pytest.mark.asyncio
async def test_pdf_agent_chunk_ordering():
    """Test that chunks are ordered by relevance"""
    pdf_agent = create_pdf_agent()
    
    response = await pdf_agent("How do options trading work?")
    
    if response and response.relevant_chunks:
        # Check that chunks are ordered by similarity score (descending)
        scores = [chunk.similarity_score for chunk in response.relevant_chunks]
        assert scores == sorted(scores, reverse=True) 