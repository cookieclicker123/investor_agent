from sentence_transformers import SentenceTransformer
from server.src.index.json_to_index import load_index, similarity_search
from langchain_community.vectorstores import FAISS as LangchainFAISS
from langchain_huggingface import HuggingFaceEmbeddings

def print_search_results(query: str, results, source: str):
    print(f"\n{'-'*80}")
    print(f"Results for query: '{query}' using {source}")
    print(f"{'-'*80}")
    
    for i, (chunk, score) in enumerate(results, 1):
        print(f"\nResult {i} (similarity score: {score:.3f}):")
        # Handle both custom DocumentChunk and Langchain Document objects
        if hasattr(chunk, 'text'):
            # Our custom DocumentChunk
            print(f"Text: {chunk.text[:400]}...")
            print(f"Source: {chunk.metadata.source_file}")
            print(f"Chunk ID: {chunk.metadata.chunk_id}/{chunk.metadata.total_chunks}")
        else:
            # Langchain Document
            print(f"Text: {chunk.page_content[:400]}...")
            print(f"Source: {chunk.metadata.get('source_file', 'Unknown')}")
            print(f"Chunk ID: {chunk.metadata.get('chunk_id', '?')}/{chunk.metadata.get('total_chunks', '?')}")

def compare_search_results():
    # Load both indexes
    custom_index_path = "server/tmp/indexes"
    langchain_index_path = "server/tmp/indexes_langchain"
    
    # Use same model name for both implementations
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    
    # Custom implementation
    index, chunks = load_index(custom_index_path)
    model = SentenceTransformer(model_name)
    
    # Langchain implementation
    embeddings = HuggingFaceEmbeddings(model_name=model_name)  # Specify the same model
    langchain_vectorstore = LangchainFAISS.load_local(
        langchain_index_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    # Test queries
    test_queries = [
        "What is the best options strategy for a bull market?",
        "How do i study the balanace sheet?"
    ]
    
    for query in test_queries:
        # Custom search
        custom_results = similarity_search(
            query=query,
            index=index,
            chunks=chunks,
            model=model,
            k=3
        )
        print_search_results(query, custom_results, "Custom Index")
        
        # Langchain search
        lc_results = langchain_vectorstore.similarity_search_with_score(query, k=3)
        lc_formatted = [(chunk, score) for chunk, score in lc_results]
        print_search_results(query, lc_formatted, "Langchain Index")
        
        print("\nPress Enter to continue to next query...")
        input()

if __name__ == "__main__":
    compare_search_results() 