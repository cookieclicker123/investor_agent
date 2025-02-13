from typing import List, Dict, Any
from server.src.data_model import DocumentChunk, ChunkMetadata

class RecursiveTextSplitter:
    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        separators: List[str] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ".", "!", "?", ";", ":", " ", ""]
        
    def split_text(self, text: str) -> List[str]:
        """Split text recursively using separators"""
        # Base case: text is short enough
        if len(text) <= self.chunk_size:
            return [text]
            
        chunks = []
        start = 0
        
        while start < len(text):
            # Find the end point for this chunk
            end = start + self.chunk_size
            
            if end >= len(text):
                chunks.append(text[start:])
                break
                
            # Try to find a natural break point
            best_end = end
            for sep in self.separators:
                # Look for separator in window around target length
                window_start = max(start, end - 100)
                window_end = min(len(text), end + 100)
                last_sep = text.rfind(sep, window_start, window_end)
                
                if last_sep != -1 and last_sep > start:
                    best_end = last_sep + len(sep)
                    break
            
            # Add chunk and move start point
            chunk = text[start:best_end].strip()
            if chunk:
                chunks.append(chunk)
            start = best_end - self.chunk_overlap
        
        return chunks

def create_document_splitters():
    """Create dense and regular splitters"""
    dense_splitter = RecursiveTextSplitter(
        chunk_size=800,
        chunk_overlap=400
    )
    
    regular_splitter = RecursiveTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    return dense_splitter, regular_splitter

def process_document(
    text: str,
    metadata: Dict[str, Any]
) -> List[DocumentChunk]:
    """Process a single document into chunks"""
    dense_splitter, regular_splitter = create_document_splitters()
    
    # Determine content type
    is_dense = any(term in text.lower() for term in [
        'financial statement',
        'balance sheet',
        'income statement'
    ])
    
    # Split text
    splitter = dense_splitter if is_dense else regular_splitter
    chunks = splitter.split_text(text)
    
    # Create DocumentChunks with metadata
    doc_chunks = []
    for i, chunk in enumerate(chunks):
        chunk_metadata = ChunkMetadata(
            **metadata,
            chunk_id=i,
            total_chunks=len(chunks),
            chunk_size=len(chunk),
            chunking_strategy="dense" if is_dense else "regular"
        )
        doc_chunks.append(DocumentChunk(
            text=chunk,
            metadata=chunk_metadata
        ))
    
    return doc_chunks
