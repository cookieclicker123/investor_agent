import asyncio
import json
import aiohttp
from typing import Callable
import uvicorn
import multiprocessing
import time
import os
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from server.src.web_app import create_web_app
from server.tests.mocks.mock_llm import create_mock_llm_client

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

def run_server():
    """Run the FastAPI server in a separate process"""
    llm_function = create_mock_llm_client(query_response=MOCK_RESPONSES)
    app = create_web_app(llm_function)
    uvicorn.run(app, host="0.0.0.0", port=8006)

async def stream_query(
    query: str, 
    on_chunk: Callable[[str], None],
    on_complete: Callable[[dict], None],
    base_url: str = "http://localhost:8006"
) -> None:
    """Stream a query to the LLM server and process the responses"""
    logger.debug(f"Sending query to {base_url}: {query}")
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                f"{base_url}/query",
                json={"query": query},
                headers={"Accept": "text/event-stream"}
            ) as response:
                logger.debug(f"Response status: {response.status}")
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    logger.debug(f"Received line: {line}")
                    if line.startswith('data: '):
                        data = json.loads(line[6:])
                        
                        if data['type'] == 'chunk':
                            logger.debug(f"Processing chunk: {data['content']}")
                            on_chunk(data['content'])
                        elif data['type'] == 'complete':
                            logger.debug(f"Processing complete response")
                            on_complete(data['content'])
        except Exception as e:
            logger.error(f"Error in stream_query: {str(e)}")
            raise

async def main():
    server_process = multiprocessing.Process(target=run_server)
    server_process.start()
    logger.info("Server started, waiting for startup...")
    time.sleep(2)  # Wait for server to start
    
    try:
        def print_chunk(chunk: str):
            print(f"\nChunk received: {chunk}")
        
        def print_complete(response: dict):
            print("\nComplete Response:")
            print(json.dumps(response, indent=2))
        
        logger.info("Starting query...")
        await stream_query(
            "How do options trading work?",
            on_chunk=print_chunk,
            on_complete=print_complete
        )
        logger.info("Query complete")
    finally:
        server_process.terminate()
        server_process.join()
        logger.info("Server shutdown complete")

if __name__ == "__main__":
    asyncio.run(main()) 