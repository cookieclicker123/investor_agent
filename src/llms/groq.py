import datetime
import time
from typing import AsyncIterator, Callable
import aiohttp
import json
import logging
import ssl
import certifi

from src.data_model import LLMRequest, LLMResponse, llmFn
from utils.config import get_groq_config

logger = logging.getLogger(__name__)

async def make_groq_request(
    model_name: str,
    prompt: str,
    api_key: str,
    temperature: float,
    max_tokens: int
) -> AsyncIterator[str]:
    """Helper function to make requests to Groq API"""
    logger.debug("Starting Groq request...")
    # Create SSL context with certifi certificates
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True,
            },
            ssl=ssl_context  # Use the SSL context instead of the cert path
        ) as response:
            logger.debug(f"Response status: {response.status}")
            if response.status != 200:
                error_text = await response.text()
                logger.error(f"Error response: {error_text}")
                raise Exception(f"Groq API error (status {response.status}): {error_text}")

            async for line in response.content:
                if line:
                    line = line.decode()
                    logger.debug(f"Raw line: {line}")
                    if line.startswith('data: '):
                        try:
                            data = json.loads(line[6:])
                            if content := data['choices'][0].get('delta', {}).get('content'):
                                logger.debug(f"Yielding content: {content}")
                                yield content
                        except (json.JSONDecodeError, KeyError, IndexError) as e:
                            logger.error(f"Error processing line: {e}")
                            continue

def create_groq_client() -> llmFn:
    """Creates a direct streaming connection to Groq API"""
    config = get_groq_config()
    
    async def generate_llm_response(
        llm_request: LLMRequest,
        on_chunk: Callable[[str], None]
    ) -> LLMResponse:
        start_time = time.time()
        chunks = []
        
        # Get the appropriate prompt based on prompt type
        prompt = (
            llm_request.prompt["selected_agent"] 
            if isinstance(llm_request.prompt, dict) 
            else llm_request.prompt
        )
        
        try:
            async for chunk in make_groq_request(
                config["model_name"],
                prompt,
                config["api_key"],
                config["temperature"],
                config["max_tokens"]
            ):
                chunks.append(chunk)
                on_chunk(chunk)  # Call on_chunk here in the main loop
                
            response = ''.join(chunks)
            
            if llm_request.as_json and response.strip():
                try:
                    response = json.loads(response)
                except json.JSONDecodeError:
                    response = {"raw_text": response}
                    
        except Exception as e:
            response = {"error": str(e)}
        
        return LLMResponse(
            generated_at=datetime.datetime.now().isoformat(),
            intent=[],  # Will be filled by groq_llm.py
            request=llm_request,
            raw_response=response,
            model_name=config["model_name"],
            model_provider=config["provider"],
            time_in_seconds=round(time.time() - start_time, 2)
        )
    
    return generate_llm_response