import datetime
import time
from typing import AsyncIterator, Callable
import aiohttp
import json
import logging
import ssl
import certifi
import asyncio

from src.data_model import LLMRequest, LLMResponse, llmFn
from utils.config import get_groq_config

logger = logging.getLogger(__name__)

async def make_groq_request(
    model_name: str,
    prompt: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    on_chunk: Callable[[str], None]
) -> AsyncIterator[str]:
    """Stream response from Groq API."""
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
            ssl=ssl_context
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Groq API error (status {response.status}): {error_text}")

            # Match Ollama's pattern exactly
            async for line in response.content:
                if line:
                    line = line.decode().strip()
                    if not line or line == 'data: [DONE]':
                        continue
                        
                    if line.startswith('data: '):
                        try:
                            data = json.loads(line[6:])
                            if content := data['choices'][0].get('delta', {}).get('content'):
                                # Match Ollama's response format
                                response_data = {"response": content}
                                if "response" in response_data:
                                    chunk = response_data["response"]
                                    on_chunk(chunk)
                                    yield chunk
                        except Exception as e:
                            logger.error(f"Error processing chunk: {e}")
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
        
        logger.debug("=== Starting Chunk Processing ===")
        logger.debug(f"Initial chunks array: {chunks}")
        
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
                config["max_tokens"],
                on_chunk
            ):
                logger.debug(f"Received chunk type: {type(chunk)}")
                logger.debug(f"Chunk content: {chunk}")
                chunks.append(chunk)
                logger.debug(f"Passing to on_chunk: {type(chunk)}, content: {chunk}")
            
            logger.debug("=== Assembling Final Response ===")
            logger.debug(f"All chunks collected: {chunks}")
            response = ''.join(chunks)
            logger.debug(f"Joined response type: {type(response)}")
            logger.debug(f"Joined response content: {response}")
            
            raw_response = {"raw_text": response}
            logger.debug(f"Wrapped response type: {type(raw_response)}")
            logger.debug(f"Wrapped response content: {raw_response}")
            
            logger.debug("=== Pydantic Validation ===")
            logger.debug(f"Data going into LLMResponse - type: {type(raw_response)}")
            logger.debug(f"Data going into LLMResponse - content: {raw_response}")
                    
            return LLMResponse(
                generated_at=datetime.datetime.now().isoformat(),
                intents=[],
                request=llm_request,
                raw_response=raw_response,
                model_name=config["model_name"],
                model_provider=config["provider"],
                time_in_seconds=round(time.time() - start_time, 2),
                confidence=0.0
            )
        except Exception as e:
            logger.error(f"Error in generate_llm_response: {str(e)}")
            return LLMResponse(
                generated_at=datetime.datetime.now().isoformat(),
                intents=[],
                request=llm_request,
                raw_response={"error": str(e)},
                model_name=config["model_name"],
                model_provider=config["provider"],
                time_in_seconds=0.0,
                confidence=0.0
            )
    
    return generate_llm_response