import logging
from typing import Optional, List
import faiss
from sentence_transformers import SentenceTransformer
from src.data_model import PDFContext, PDFAgentResponse
from src.index.json_to_index import load_index
import numpy as np

logger = logging.getLogger(__name__)

# Disable external logging
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
logging.getLogger('faiss').setLevel(logging.WARNING)

def initialize_embeddings(model_name: str = 'all-MiniLM-L6-v2', device: str = 'mps') -> SentenceTransformer:
    """Initialize the embedding model"""
    try:
        return SentenceTransformer(f'sentence-transformers/{model_name}', device=device)
    except Exception as e:
        logger.error(f"Failed to initialize embedding model: {str(e)}")
        raise

def load_faiss_index(index_path: str) -> faiss.Index:
    """Load the FAISS index"""
    try:
        return faiss.read_index(index_path)
    except Exception as e:
        logger.error(f"Failed to load FAISS index: {str(e)}")
        raise

def create_pdf_context(
    text: str,
    source_file: str,
    chunk_id: int,
    total_chunks: int,
    similarity_score: float
) -> PDFContext:
    """Create a PDFContext object from chunk data"""
    return PDFContext(
        text=text,
        source_file=source_file,
        chunk_id=chunk_id,
        total_chunks=total_chunks,
        similarity_score=similarity_score
    )

async def get_relevant_chunks(query: str, index_path: str) -> List[PDFContext]:
    """Get relevant chunks from FAISS index"""
    try:
        # Validate query
        if not query or query.isspace():
            logger.warning("Empty or whitespace-only query received")
            return []
            
        embedding_model = initialize_embeddings()
        
        # Load both index and chunks
        index, chunks = load_index(index_path)
        
        # Generate query embedding
        query_embedding = embedding_model.encode([query])[0]
        
        # Search FAISS index
        k = 7  # Number of chunks to retrieve
        distances, indices = index.search(query_embedding.reshape(1, -1), k)
        
        # Convert distances to similarity scores
        # Using exponential normalization for better score distribution
        scores = [float(np.exp(-d)) for d in distances[0]]
        
        # Sort chunks by score
        chunk_scores = list(zip(indices[0], scores))
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Create PDFContext objects using the actual chunks
        pdf_chunks = []
        for i, (idx, score) in enumerate(chunk_scores):
            chunk = chunks[int(idx)]
            pdf_chunks.append(PDFContext(
                text=chunk.text,
                source_file=chunk.metadata.source_file,
                chunk_id=chunk.metadata.chunk_id,
                total_chunks=chunk.metadata.total_chunks,
                similarity_score=score
            ))
        
        return pdf_chunks
        
    except Exception as e:
        logger.error(f"Error retrieving chunks: {str(e)}")
        return []

async def get_pdf_context(query: str, index_path: str) -> Optional[PDFAgentResponse]:
    """Factory function to get PDF context"""
    try:
        relevant_chunks = await get_relevant_chunks(query, index_path)
        
        if not relevant_chunks:
            logger.warning(f"No relevant chunks found for query: {query}")
            return None
            
        return PDFAgentResponse(
            relevant_chunks=relevant_chunks,
            synthesized_answer=None  # This will be filled by the PDF agent
        )
        
    except Exception as e:
        logger.error(f"Error creating PDF agent response: {str(e)}")
        return None
