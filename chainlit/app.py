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
    """Define starter questions for investor queries."""
    return [
        cl.Starter(
            label="TSLA Market Sentiment",
            message="What is the current market sentiment on TSLA?",
            icon="/public/production_2.svg",
        ),
        cl.Starter(
            label="TSLA Earnings Details",
            message="Show me the latest earnings report details for TSLA.",
            icon="/public/Tanooki.svg",
        ),
        cl.Starter(
            label="TSLA Options Strategies",
            message="How do advanced options strategies work for TSLA?",
            icon="/public/TV.svg",
        ),
        cl.Starter(
            label="TSLA Competitive Edge",
            message="Summarize the key competitive advantages of TSLA.",
            icon="/public/production.svg",
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

    # Single async function to handle message creation and animation
    async def show_loading_message():
        msg = cl.Message(content="**📊Processing query**")
        await msg.send()
        print("Message created with loading animation starting")  # Debug print
        
        try:
            while True:
                for i in range(4):
                    msg.content = "**📊Processing query**" + "." * i
                    await msg.update()
                    await asyncio.sleep(0.25)
        except asyncio.CancelledError:
            print("Loading animation cancelled")  # Debug print
            return msg

    # Start the loading message and get the message object
    loading_task = asyncio.create_task(show_loading_message())
    msg = await loading_task

    try:
        endpoint = f"/{MODEL}/query"
        with client.stream('POST', endpoint, json={"query": message.content}) as response:
            if response.status_code != 200:
                loading_task.cancel()
                await msg.update(content=f"Error: {response.status_code}")
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
                            loading_task.cancel()
                            first_chunk = False
                        await msg.stream_token(data['content'])
                    elif data['type'] == 'complete':
                        complete_response = data['content']
                        async with cl.Step(name="Complete Response") as step:
                            formatted_output = f"```json\n{json.dumps(complete_response, indent=2)}\n```"
                            step.output = formatted_output
                        
                except Exception as e:
                    print(f"Error processing chunk: {e}")
                            
    except Exception as e:
        print(f"Error in main handler: {e}")
        await msg.update(content=f"Error: {str(e)}")
    finally:
        if not loading_task.done():
            loading_task.cancel() 