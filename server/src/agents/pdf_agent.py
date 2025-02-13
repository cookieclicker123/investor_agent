import logging
from typing import Optional
from server.src.data_model import PDFAgentResponse, pdfAgentFn
from server.src.tools.pdf_tools import get_pdf_context
from server.src.prompts.prompts import PDF_AGENT_PROMPT

logger = logging.getLogger(__name__)

def create_pdf_agent() -> pdfAgentFn:
    """Factory function to create PDF agent functionality"""
    
    # Initialize index path at creation time
    index_path = "server/tmp/indexes"
    logger.debug(f"Initializing PDF agent with index: {index_path}")
    
    async def process_query(query: str) -> Optional[PDFAgentResponse]:
        """Process a PDF-related query using the index"""
        try:
            logger.debug(f"Processing PDF query: {query}")
            
            # Return None for empty queries
            if not query or query.isspace():
                return None
            
            context = await get_pdf_context(query, index_path)
            
            if context and context.relevant_chunks:
                # Format chunks for prompt
                formatted_chunks = []
                for chunk in context.relevant_chunks:
                    formatted_chunks.append(
                        f"Source: {chunk.source_file}\n"
                        f"Relevance: {chunk.similarity_score:.2f}\n"
                        f"Content: {chunk.text}\n"
                    )
                
                # Add formatted context to prompt
                context.synthesized_answer = PDF_AGENT_PROMPT.format(
                    context="\n".join(formatted_chunks),
                    query=query,
                    pdf_history=""
                )
                
                return context
            
            return None
            
        except Exception as e:
            logger.error(f"Error in PDF agent processing: {str(e)}")
            return None
    
    return process_query
