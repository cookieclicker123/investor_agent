from server.src.data_model import PDFContext, PDFAgentResponse

def get_dummy_context(query: str) -> PDFAgentResponse:
    """Return mock PDF context for testing"""
    dummy_chunks = [
        PDFContext(
            text="[From CME Options Trading Guide PDF] Chapter 3: Options Trading Strategies - "
                 "This section covers the most effective options trading strategies including "
                 "covered calls, protective puts, and spread strategies...",
            source_file="CME_Options_Trading_Guide.pdf",
            chunk_id=1,
            total_chunks=3,
            similarity_score=0.85
        ),
        PDFContext(
            text="[From Options Strategy Handbook PDF] Section 2.1: Market Analysis - "
                 "When selecting an options strategy, first analyze market conditions and "
                 "volatility expectations...",
            source_file="Options_Strategy_Handbook.pdf",
            chunk_id=2,
            total_chunks=4,
            similarity_score=0.75
        )
    ]
    
    return PDFAgentResponse(
        relevant_chunks=dummy_chunks,
        synthesized_answer=None  # Will be filled by LLM
    )


def query_documents(query: str) -> PDFAgentResponse:
    """Mock implementation of document querying"""
    return get_dummy_context(query)
