from src.data_model import PDFAgentResponse, LLMResponse
from .pdf_dummy_tools import query_documents

class PDFDummyAgent:
    def process_query(self, query: str) -> PDFAgentResponse:
        """Process a PDF-related query with mock data"""
        # Get mock context
        context = query_documents(query)
        return context

    def format_for_llm(self, context: PDFAgentResponse) -> str:
        """Format PDF context for LLM consumption"""
        formatted_chunks = []
        for chunk in context.relevant_chunks:
            formatted_chunks.append(
                f"Source: {chunk.source_file}\n"
                f"Relevance: {chunk.similarity_score}\n"
                f"Content: {chunk.text}\n"
            )
        return "\n".join(formatted_chunks)
