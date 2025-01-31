import os
import json
import asyncio
import argparse
from src.ollama_llm import create_ollama_llm
from src.groq_llm import create_groq_llm
from tests.mocks.mock_llm import create_mock_llm_client
from src.data_model import LLMRequest
from utils.config import get_ollama_config, get_groq_config

async def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run Financial Agent with LLM')
    
    # Get available models from config
    ollama_config = get_ollama_config()
    groq_config = get_groq_config()

    ollama_models = [ollama_config["model_name"]]
    groq_models = [groq_config["model_name"]]

    print(f"Available Ollama models: {ollama_models}")
    print(f"Available Groq models: {groq_models}")
    
    parser.add_argument(
        '--model', 
        default=ollama_config["model_name"],
        choices=ollama_models + groq_models,
        help=f'Model to use (default: {ollama_config["model_name"]})'
    )
    
    parser.add_argument(
        '--mock',
        action='store_true',
        help='Use mock LLM instead of real models'
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
                if "conversations" not in conversation_history:
                    conversation_history = {"conversations": []}
        else:
            conversation_history = {"conversations": []}
    except json.JSONDecodeError:
        conversation_history = {"conversations": []}
    
    # Create LLM based on args
    if args.mock:
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
        # Choose between Ollama and Groq based on model name
        if args.model in ollama_models:
            llm = create_ollama_llm()
            print(f"\nUsing Ollama LLM with model: {args.model}")
        else:
            llm = create_groq_llm()
            print(f"\nUsing Groq LLM with model: {args.model}")
    
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
