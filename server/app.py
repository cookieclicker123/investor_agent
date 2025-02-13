import os
import json
import asyncio
import argparse
import logging
from src.ollama_llm import create_ollama_llm
from src.groq_llm import create_groq_llm
from tests.mocks.mock_llm import create_mock_llm_client
from src.data_model import LLMRequest
import uvicorn
from src.web_app import create_web_app
import threading

# Simple file logging
logging.basicConfig(
    filename='debug.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Simple mock responses
MOCK_RESPONSES = {
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

def run_web_server(llm):
    """Run the FastAPI server in a separate thread"""
    app = create_web_app(llm)
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8006,
        log_level="info"
    )

async def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run Financial Agent with LLM')
    
    # Define available models
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    groq_model = os.getenv("GROQ_MODEL_NAME", "deepseek-r1-distill-llama-70b")
    
    parser.add_argument(
        '--model', 
        default=ollama_model,
        choices=[ollama_model, groq_model],
        help=f'Model to use (default: {ollama_model})'
    )
    
    parser.add_argument(
        '--mock',
        action='store_true',
        help='Use mock LLM instead of real models'
    )

    args = parser.parse_args()
    
    # Ensure the tmp directory exists
    os.makedirs("./server/tmp", exist_ok=True)
    
    # Load existing conversation history or create new one
    history_file = "./server/tmp/conversation_log.json"
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
        llm = create_mock_llm_client(query_response=MOCK_RESPONSES)
        print("\nUsing Mock LLM")
    else:
        if args.model == ollama_model:
            llm = create_ollama_llm()
            print(f"\nUsing Ollama LLM with model: {args.model}")
        else:
            llm = create_groq_llm()
            print(f"\nUsing Groq LLM with model: {args.model}")

    # Start web server in a separate thread
    print("\nStarting web server...")
    server_thread = threading.Thread(target=run_web_server, args=(llm,))
    server_thread.daemon = True  # This ensures the thread will be killed when the main program exits
    server_thread.start()
    
    # Give the server a moment to start
    await asyncio.sleep(2)
    
    # Modified streaming callback
    def on_chunk(chunk: str):
        logger.debug("=== CHUNK PROCESSING ===")
        logger.debug(f"1. Original chunk: '{chunk}'")
        
        # Only remove think tags, preserve all other whitespace
        cleaned = chunk.replace("<think>", "").replace("</think>", "")
        logger.debug(f"2. After tag removal: '{cleaned}'")
        
        # Don't strip! We need the spaces between words
        if cleaned:
            logger.debug(f"3. Printing chunk: '{cleaned}'")
            print(cleaned, end="", flush=True)
        else:
            logger.debug("3. Empty chunk, skipping")
    
    print("\nFinancial Agent Chat Interface")
    print("Type 'exit' to quit")
    print("\nExample queries:")
    print("- What is the current price of (AAPL)?")
    print("- How do options trading work?")
    print("- What's happening in the market today?\n")
    
    while True:
        query = input("\nEnter your query: ").strip()
        
        if query.lower() == 'exit':
            break
            
        if not query:
            continue
            
        print("\n" + "-" * 50)
        
        request = LLMRequest(
            query=query,
            prompt="",
            as_json=True
        )
        
        try:
            # Get final response
            logger.debug("=== Before Final LLM Response ===")
            logger.debug(f"Request prompt type: {type(request.prompt)}")
            logger.debug(f"Request prompt content: {request.prompt}")

            response = await llm(request, on_chunk)

            logger.debug("=== After LLM Response ===")
            logger.debug(f"Initial response type: {type(response.raw_response)}")
            logger.debug(f"Initial response content: {response.raw_response}")
            
            print("\n\nIntents detected:", response.intents)
            print("-" * 50)
            
            # Handle the raw response based on its type
            if isinstance(response.raw_response, dict):
                chunks = [response.raw_response]
            else:
                chunks = [{
                    "meta_agent": {"raw_text": ""},
                    "raw_text": response.raw_response
                }]
            
            # Create conversation entry
            conversation_entry = {
                "request": request.model_dump(exclude_none=True),
                "response": response.model_dump(exclude_none=True),
                "chunks": chunks
            }
            
            # Add to history
            conversation_history["conversations"].append(conversation_entry)
            
            # Save updated history
            with open(history_file, "w") as f:
                json.dump(conversation_history, f, indent=2)
            
            print("\nConversation logged to ./server/tmp/conversation_log.json")
            
        except Exception as e:
            logger.error("Error processing response", exc_info=True)
            print(f"\nError processing response: {e}")
            continue

if __name__ == "__main__":
    asyncio.run(main())
