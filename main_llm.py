import os
import json
import asyncio
import argparse
from src.ollama_llm import create_ollama_llm
from src.mock_llm import create_mock_llm_client
from src.data_model import LLMRequest
from utils.config import get_ollama_config

async def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run Financial Agent with LLM')
    
    # Get available models from config
    config = get_ollama_config()
    
    parser.add_argument(
        '--model', 
        default="llama3.2:3b",
        help='Ollama model to use (default: llama3.2:3b)'
    )
    
    parser.add_argument(
        '--mock',
        action='store_true',
        help='Use mock LLM instead of real Ollama'
    )
    
    args = parser.parse_args()
    
    # Ensure the tmp directory exists
    os.makedirs("tmp", exist_ok=True)
    
    # Load existing conversation history or create new one
    history_file = "tmp/conversation_log.json"
    try:
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                conversation_history = json.load(f)
                # Ensure the conversations key exists
                if "conversations" not in conversation_history:
                    conversation_history = {"conversations": []}
        else:
            conversation_history = {"conversations": []}
    except json.JSONDecodeError:
        # Handle case where file exists but is invalid JSON
        conversation_history = {"conversations": []}
    
    # Create LLM based on args
    if args.mock:
        # Define mock responses for each query type
        mock_responses = {
            "What is the current price of (AAPL)?": {
                "answer": "The current price of AAPL is $185.64"
            },
            "How do options trading work?": {
                "answer": "Options contracts are financial instruments that give the holder the right, but not the obligation, to buy or sell an asset at a predetermined price within a specific time frame"
            },
            "What's happening in the market today?": {
                "answer": "Today's market is showing mixed signals with tech stocks leading gains while energy sector faces pressure"
            }
        }
        llm = create_mock_llm_client(query_response=mock_responses)
        print("\nUsing Mock LLM")
    else:
        llm = create_ollama_llm()
        print(f"\nUsing Ollama LLM with model: {args.model}")
    
    # Simple streaming callback
    def on_chunk(chunk: str):
        print(chunk, end="", flush=True)
    
    print("\nFinancial Agent Chat Interface")
    print("Type 'exit' to quit")
    print("\nExample queries:")
    print("- What is the current price of (AAPL)?")
    print("- How do options trading work?")
    print("- What's happening in the market today?\n")
    
    while True:
        # Get user input
        query = input("\nEnter your query: ").strip()
        
        if query.lower() == 'exit':
            break
            
        if not query:
            continue
            
        print("\n" + "-" * 50)
        
        request = LLMRequest(
            query=query,
            prompt="",  # Will be filled by the LLM
            as_json=True
        )
        
        response = await llm(request, on_chunk)
        print("\n\nIntent detected:", response.intent)
        print("-" * 50)
        
        # Create conversation entry
        conversation_entry = {
            "request": request.model_dump(),
            "response": response.model_dump(),
            "chunks": [response.raw_response]
        }
        
        # Add to history
        conversation_history["conversations"].append(conversation_entry)
        
        # Save updated history
        with open(history_file, "w") as f:
            json.dump(conversation_history, f, indent=2)
        
        print("\nConversation logged to tmp/conversation_log.json")

if __name__ == "__main__":
    asyncio.run(main())
