import os
import json
import asyncio
from src.data_model import LLMRequest
from src.mock_llm import create_mock_llm_client

async def main():
    # Define the mock query response
    query_response = {
        "what are options": {"answer": "Options contracts are financial instruments that give the holder the right, but not the obligation, to buy or sell an asset at a predetermined price within a specific time frame"},
        "what impact is deepseek ai's new set of models having on the us stock market this week?": {"answer": "DeepSeek AI's new set of models have had a positive impact on the US stock market this week, with the company's stock price increasing by 10% due to the release of its new models"},
        "tell me the performance metrics of pltr in the stock market this week": {"answer": "It has performed very well because palentir will benefit from the open sourcing of models like deepseek-r1 to use as the base model in its own systems due to the open weights nature of the radical and innovative new model"}
    }
    
    # Create a mock LLM client
    generate_response = create_mock_llm_client(query_response)
    
    # Ensure the tmp directory exists
    os.makedirs("tmp", exist_ok=True)
    
    try:
        while True:
            # Get user input
            user_input = input("You: ").strip().lower()
            
            # Check for exit conditions
            if user_input in ["quit", "exit"]:
                print("Exiting chat. Goodbye!")
                break
            
            # Create a fake LLM request
            request = LLMRequest(query=user_input, prompt="", as_json=True)
            
            # Define a simple on_chunk function
            def on_chunk(chunk: str):
                print(chunk, end='', flush=True)
            
            # Generate a response
            try:
                response = await generate_response(request, on_chunk)
            except Exception as e:
                print(f"\nError: {e}")
                continue
            
            # Extract and print only the answer
            answer = response.raw_response.get("answer", "")
            
            # Log the conversation to a JSON file
            conversation_log = {
                "request": request.model_dump(),
                "response": response.model_dump(),
                "chunks": [answer]  # Log the full answer instead of individual chunks
            }
            
            with open("tmp/conversation_log.json", "w") as log_file:
                json.dump(conversation_log, log_file, indent=2)
            
            print("\n\n")
            print("Conversation logged to tmp/conversation_log.json")
    
    except KeyboardInterrupt:
        print("\nExiting chat. Goodbye!")

if __name__ == "__main__":
    asyncio.run(main())