import sys
import os
from pathlib import Path
import asyncio

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

@cl.set_starters
async def starters():
    """Define starter questions across various topics."""
    return [
        cl.Starter(
            label="Crypto Analysis",
            message="How is Bitcoin performing against other cryptocurrencies?",
            icon="/public/money.svg",
        ),
        cl.Starter(
            label="Options Basics",
            message="Explain how put options work with a simple example.",
            icon="/public/report.svg",
        ),
        cl.Starter(
            label="Tech Sector",
            message="Compare the AI strategies of NVIDIA, MSFT, and GOOGL.",
            icon="/public/rocket.svg",
        ),
        cl.Starter(
            label="Economic Indicators",
            message="What do the latest inflation numbers mean for interest rates?",
            icon="/public/chart.svg",
        )
    ]

# Use @cl.set_starters for UI starters
@cl.set_starters
async def set_question_starters():
    print("Setting starters")  # Debugging line
    return await starters()

@cl.on_chat_start
async def start():
    """Initialize chat session"""
    health = client.get("/health")
    health_data = health.json()
    cl.user_session.set("messages", [])

@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages"""
    print(f"Received message: {message.content}")

    # Create loading message
    loading_msg = cl.Message(content="**📊Processing query**")
    await loading_msg.send()
    print("Loading message created")

    # Create separate response message
    response_msg = cl.Message(content="")
    await response_msg.send()
    print("Response message created")

    # Animation task for loading message
    async def animate_loading():
        try:
            while True:
                for i in range(4):
                    loading_msg.content = "**📊Processing query**" + "." * i
                    await loading_msg.update()
                    await asyncio.sleep(0.25)
        except asyncio.CancelledError:
            print("Animation cancelled")
            await loading_msg.remove()

    # Start animation without awaiting
    loading_task = asyncio.create_task(animate_loading())
    print("Animation started")

    try:
        endpoint = f"/{MODEL}/query"
        with client.stream('POST', endpoint, json={"query": message.content}) as response:
            if response.status_code != 200:
                loading_task.cancel()
                await response_msg.update(content=f"Error: {response.status_code}")
                return

            complete_response = {}
            first_chunk = True
            
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
                        if first_chunk:
                            loading_task.cancel()  # Stop animation when response starts
                            first_chunk = False
                        await response_msg.stream_token(data['content'])
                    elif data['type'] == 'complete':
                        complete_response = data['content']
                        async with cl.Step(name="Complete Response") as step:
                            formatted_output = f"```json\n{json.dumps(complete_response, indent=2)}\n```"
                            step.output = formatted_output
                        
                except Exception as e:
                    print(f"Error processing chunk: {e}")
                            
    except Exception as e:
        print(f"Error in main handler: {e}")
        await response_msg.update(content=f"Error: {str(e)}")
    finally:
        if not loading_task.done():
            loading_task.cancel() 