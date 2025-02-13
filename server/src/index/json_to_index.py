import os
import json
import numpy as np
import faiss
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from .document_processor import process_document
from server.src.data_model import DocumentChunk

def load_and_split_texts(text_folder: str) -> List[DocumentChunk]:
    """Load JSON files and split into chunks"""
    chunks = []
    
    for file_name in os.listdir(text_folder):
        if file_name.endswith('.json'):
            with open(os.path.join(text_folder, file_name), "r") as file:
                data = json.load(file)
                text = data["text"]
                metadata = data["metadata"]
                
                # Process document into chunks
                doc_chunks = process_document(text, metadata)
                chunks.extend(doc_chunks)
    
    return chunks

def create_faiss_index(
    text_folder: str,
    index_path: str,
    embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2'
) -> None:
    """Create and save FAISS index from document chunks"""
    # Load and split documents
    chunks = load_and_split_texts(text_folder)
    
    # Initialize embedding model
    model = SentenceTransformer(embedding_model)
    
    # Create embeddings
    texts = [chunk.text for chunk in chunks]
    embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
    embeddings = embeddings.cpu().numpy()  # Convert to numpy array
    
    # Normalize vectors for cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Create FAISS index
    dimension = embeddings.shape[1]  # Get embedding dimension
    index = faiss.IndexFlatIP(dimension)  # Inner product = cosine similarity for normalized vectors
    index.add(embeddings)
    
    # Create output directory if it doesn't exist
    os.makedirs(index_path, exist_ok=True)
    
    # Save FAISS index
    faiss.write_index(index, os.path.join(index_path, "faiss.index"))
    
    # Save chunks separately (FAISS only stores vectors)
    chunks_data = [
        {
            "text": chunk.text,
            "metadata": chunk.metadata.model_dump()
        } for chunk in chunks
    ]
    
    with open(os.path.join(index_path, "chunks.json"), "w") as f:
        json.dump(chunks_data, f)
    
    print(f"Created FAISS index with {len(chunks)} chunks")
    print(f"Index saved to {index_path}")

def load_index(
    index_path: str
) -> Tuple[faiss.Index, List[DocumentChunk]]:
    """Load saved index and chunks"""
    # Load FAISS index
    index = faiss.read_index(os.path.join(index_path, "faiss.index"))
    
    # Load chunks
    with open(os.path.join(index_path, "chunks.json"), "r") as f:
        chunks_data = json.load(f)
        chunks = [
            DocumentChunk(
                text=c["text"],
                metadata=c["metadata"]
            ) for c in chunks_data
        ]
    
    return index, chunks

def similarity_search(
    query: str,
    index: faiss.Index,
    chunks: List[DocumentChunk],
    model: SentenceTransformer,
    k: int = 4
) -> List[Tuple[DocumentChunk, float]]:
    """Search for similar chunks"""
    # Create query embedding
    query_embedding = model.encode([query], convert_to_tensor=True)
    query_embedding = query_embedding.cpu().numpy()
    
    # Normalize query vector
    faiss.normalize_L2(query_embedding)
    
    # Search
    scores, indices = index.search(query_embedding, k)
    
    # Return chunks with scores
    results = [
        (chunks[int(idx)], float(score))  # Convert idx to int for list indexing
        for score, idx in zip(scores[0], indices[0])
        if idx >= 0  # Ensure no negative indices
    ]
    
    return results

if __name__ == "__main__":
    text_folder = "./server/tmp/processed"
    index_path = "./server/tmp/indexes"
    create_faiss_index(text_folder, index_path)
