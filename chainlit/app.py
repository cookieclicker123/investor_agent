import sys
import os
from pathlib import Path

# Add root directory to Python path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

import chainlit as cl
import json
from src.server import app
from fastapi.testclient import TestClient

# Get model from environment variable with validation
MODEL = os.getenv("MODEL", "groq").lower()
if MODEL not in ["groq", "ollama"]:
    raise ValueError(f"Invalid MODEL environment variable. Must be 'groq' or 'ollama', got '{MODEL}'")

# Create client from our configured FastAPI server
client = TestClient(app)

@cl.on_chat_start
async def start():
    """Initialize chat session"""
    # Test server health and get available models
    health = client.get("/health")
    health_data = health.json()
    
    print("\n=== Investor Agent Chat ===")
    print(f"Server Status: {health_data['status']}")
    print(f"Using: {MODEL.upper()}")
    print(f"Model: {health_data['models'][MODEL]}")
    print("==========================\n")
    
    cl.user_session.set("messages", [])

@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages"""
    endpoint = f"/{MODEL}/query"
    
    msg = cl.Message(content="")
    await msg.send()

    try:
        with client.stream('POST', endpoint, json={"query": message.content}) as response:
            if response.status_code != 200:
                await msg.update(content=f"Error: {response.status_code}")
                return

            complete_response = {}  # Store complete response
            for line in response.iter_lines():
                if not line:
                    continue
                    
                try:
                    if isinstance(line, bytes):
                        line = line.decode('utf-8')
                    
                    if line.startswith('data: '):
                        line = line.replace('data: ', '')
                    
                    data = json.loads(line)
                    if data['type'] == 'chunk':
                        await msg.stream_token(data['content'])
                    elif data['type'] == 'complete':
                        # Store complete response data
                        complete_response = data['content']
                        print("\n=== Complete Response ===")
                        print(json.dumps(complete_response, indent=2))
                        print("========================\n")
                        
                except json.JSONDecodeError as e:
                    print(f"JSON decode error: {e} for line: {line}")
                except Exception as e:
                    print(f"Error processing chunk: {e}")
                            
    except Exception as e:
        print(f"Error in main handler: {e}")
        await msg.update(content=f"Error: {str(e)}") 