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
            logger.debug(f"Response status: {response.status}")
            if response.status != 200:
                error_text = await response.text()
                logger.error(f"Error response: {error_text}")
                raise Exception(f"Groq API error (status {response.status}): {error_text}")

            async for line in response.content:
                if line:
                    line = line.decode().strip()
                    logger.debug(f"Raw line: {line}")
                    
                    # Skip empty lines and [DONE] message
                    if not line or line == 'data: [DONE]':
                        continue
                        
                    if line.startswith('data: '):
                        try:
                            data = json.loads(line[6:])
                            if content := data['choices'][0].get('delta', {}).get('content'):
                                logger.debug(f"Yielding content: {content}")
                                yield content
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON decode error on line: {line}")
                            continue
                        except (KeyError, IndexError) as e:
                            logger.error(f"Error processing line structure: {e}")
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
        
        # Add logging before chunks processing
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
                config["max_tokens"]
            ):
                logger.debug(f"Received chunk type: {type(chunk)}")
                logger.debug(f"Chunk content: {chunk}")
                chunks.append(chunk)
                logger.debug(f"Passing to on_chunk: {type(chunk)}, content: {chunk}")
                on_chunk(chunk)
            
            # Log the assembled response
            logger.debug("=== Assembling Final Response ===")
            logger.debug(f"All chunks collected: {chunks}")
            response = ''.join(chunks)
            logger.debug(f"Joined response type: {type(response)}")
            logger.debug(f"Joined response content: {response}")
            
            # Log the wrapping in raw_text dict
            raw_response = {"raw_text": response}
            logger.debug(f"Wrapped response type: {type(raw_response)}")
            logger.debug(f"Wrapped response content: {raw_response}")
            
            # Add Pydantic validation debugging
            logger.debug("=== Pydantic Validation ===")
            logger.debug(f"Data going into LLMResponse - type: {type(raw_response)}")
            logger.debug(f"Data going into LLMResponse - content: {raw_response}")
                    
            return LLMResponse(
                generated_at=datetime.datetime.now().isoformat(),
                intent=[],  # Will be filled by groq_llm.py
                request=llm_request,
                raw_response=raw_response,  # Always a dict now
                model_name=config["model_name"],
                model_provider=config["provider"],
                time_in_seconds=round(time.time() - start_time, 2)
            )
        except Exception as e:
            logger.error(f"Error in generate_llm_response: {str(e)}")
            raw_response = {"error": str(e)}
            raise
    
    return generate_llm_response