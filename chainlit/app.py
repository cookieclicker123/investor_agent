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
    
    # Create empty message first
    msg = cl.Message(content="")
    await msg.send()

    try:
        # Send query to server - use message.content
        with client.stream('POST', endpoint, json={"query": message.content}) as response:
            if response.status_code != 200:
                await msg.update(content=f"Error: {response.status_code}")
                return

            complete_response = {}
            # Process streaming response
            for line in response.iter_lines():
                if not line:
                    continue
                    
                try:
                    if isinstance(line, bytes):
                        line = line.decode('utf-8')
                    
                    if line.startswith('data: '):
                        line = line.replace('data: ', '')
                    
                    data = json.loads(line)
                    print(f"\nReceived data type: {data.get('type')}")  # Debug print
                    
                    if data['type'] == 'chunk':
                        await msg.stream_token(data['content'])
                    elif data['type'] == 'complete':
                        print("\n=== Received complete response ===")  # Debug print
                        complete_response = data['content']
                        
                        # Debug prints for response content
                        print(f"Response type: {type(complete_response)}")
                        print(f"Response keys: {complete_response.keys() if isinstance(complete_response, dict) else 'Not a dict'}")
                        
                        try:
                            # Create step using context manager instead
                            print("\nCreating step with context manager...")  # Debug print
                            async with cl.Step(name="Complete Response") as step:
                                print("Setting step output...")  # Debug print
                                formatted_json = json.dumps(complete_response, indent=2)
                                step.output = formatted_json
                                print("Step output set")  # Debug print
                            
                        except Exception as step_error:
                            print(f"\nError in step creation: {step_error}")
                            print(f"Error type: {type(step_error)}")
                        
                except json.JSONDecodeError as e:
                    print(f"JSON decode error: {e} for line: {line}")
                except Exception as e:
                    print(f"Error processing chunk: {e}")
                    print(f"Error type: {type(e)}")  # Debug print
                            
    except Exception as e:
        print(f"Error in main handler: {e}")
        await msg.update(content=f"Error: {str(e)}") 