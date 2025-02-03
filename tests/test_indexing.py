import pytest
import os
import json
import tempfile
import shutil
from pathlib import Path
from src.index.document_processor import RecursiveTextSplitter, process_document
from src.index.json_to_index import create_faiss_index, load_index, similarity_search
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS as LangchainFAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

@pytest.fixture
def sample_text():
    return """
    Pattern evolution:
    HOW TO USE THIS BOOKLET
    Each illustration demonstrates the effect of time decay on 
    the total option premium involved in the position. The left 
    vertical axis shows the profit/loss scale.
    
    The horizontal zero line in the middle is the break-even point, 
    not including commissions. Therefore, anything above that line 
    indicates profits, anything below it, losses.
    
    For more information, go to cmegroup.com/options
    """

@pytest.fixture
def sample_metadata():
    return {
        "title": "Test Options Guide",
        "author": "Test Author",
        "creation_date": "2024-01-01",
        "source_file": "test.pdf"
    }

@pytest.fixture
def test_dirs():
    """Create temporary directories for testing"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        processed_dir = Path(tmp_dir) / "processed"
        index_dir = Path(tmp_dir) / "indexes"
        processed_dir.mkdir()
        index_dir.mkdir()
        
        # Create a sample JSON file
        sample_data = {
            "text": "Sample text for testing vector stores",
            "metadata": {
                "title": "Test",
                "author": "Test",
                "creation_date": "2024",
                "source_file": "test.pdf"
            }
        }
        with open(processed_dir / "test.json", "w") as f:
            json.dump(sample_data, f)
            
        yield str(processed_dir), str(index_dir)

def test_text_splitter_basic(sample_text):
    """Test basic text splitting functionality"""
    # Using more realistic chunk sizes matching our actual implementation
    splitter = RecursiveTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(sample_text)
    
    assert len(chunks) > 0  # Should produce at least one chunk
    assert all(len(chunk) <= 1000 for chunk in chunks)  # Chunks should respect max size
    
    # Test chunk overlap
    if len(chunks) > 1:
        # Check for some text overlap between consecutive chunks
        for i in range(len(chunks)-1):
            current_chunk = chunks[i]
            next_chunk = chunks[i+1]
            # There should be some common text between chunks
            assert any(sent in next_chunk for sent in current_chunk.split('.') if sent)

def test_document_processing(sample_text, sample_metadata):
    """Test document processing with metadata"""
    chunks = process_document(sample_text, sample_metadata)
    
    assert len(chunks) > 0
    assert all(hasattr(chunk, 'text') for chunk in chunks)
    assert all(hasattr(chunk, 'metadata') for chunk in chunks)
    assert all(chunk.metadata.chunk_id < chunk.metadata.total_chunks 
              for chunk in chunks)

def test_index_creation_and_loading(test_dirs):
    """Test FAISS index creation and loading"""
    processed_dir, index_dir = test_dirs
    
    # Create index
    create_faiss_index(processed_dir, index_dir)
    
    # Verify files exist
    assert os.path.exists(os.path.join(index_dir, "faiss.index"))
    assert os.path.exists(os.path.join(index_dir, "chunks.json"))
    
    # Test loading
    index, chunks = load_index(index_dir)
    assert index is not None
    assert len(chunks) > 0

def test_similarity_search(test_dirs):
    """Test similarity search functionality"""
    processed_dir, index_dir = test_dirs
    
    # Create index
    create_faiss_index(processed_dir, index_dir)
    index, chunks = load_index(index_dir)
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # Test search
    query = "What is the profit/loss scale?"
    results = similarity_search(query, index, chunks, model, k=2)
    
    assert len(results) <= 2
    assert all(isinstance(score, float) for _, score in results)
    assert all(0 <= score <= 1 for _, score in results)

def test_compare_implementations(sample_text, sample_metadata, test_dirs):
    """Compare custom vs Langchain implementations"""
    processed_dir, index_dir = test_dirs
    
    # Custom implementation
    custom_chunks = process_document(sample_text, sample_metadata)
    
    # Langchain implementation
    lc_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    lc_chunks = lc_splitter.split_text(sample_text)
    
    # Compare number of chunks (should be similar)
    assert abs(len(custom_chunks) - len(lc_chunks)) <= 1
    
    # Compare chunk sizes
    custom_sizes = [len(chunk.text) for chunk in custom_chunks]
    lc_sizes = [len(chunk) for chunk in lc_chunks]
    avg_custom = sum(custom_sizes) / len(custom_sizes)
    avg_lc = sum(lc_sizes) / len(lc_sizes)
    
    # Average chunk sizes should be within 20% of each other
    assert abs(avg_custom - avg_lc) / avg_lc < 0.2

def test_metadata_preservation(test_dirs):
    """Test metadata is preserved through the pipeline"""
    processed_dir, index_dir = test_dirs
    
    # Create and load index
    create_faiss_index(processed_dir, index_dir)
    index, chunks = load_index(index_dir)
    
    # Check metadata
    assert all(chunk.metadata.title for chunk in chunks)
    assert all(chunk.metadata.author for chunk in chunks)
    assert all(chunk.metadata.source_file for chunk in chunks) 