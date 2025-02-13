import datetime
import time
import json
import aiohttp
import logging
from typing import AsyncIterator, Callable
from src.data_model import LLMRequest, LLMResponse, OnTextFn, llmFn

logger = logging.getLogger(__name__)

async def _stream_ollama_response(
    url: str,
    model_name: str,
    prompt: str,
    on_chunk: OnTextFn
) -> AsyncIterator[str]:
    """Stream response from Ollama API."""
    logger.debug("Starting Ollama request...")
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            url,
            json={"model": model_name, "prompt": prompt, "stream": True}
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                logger.error(f"Error response: {error_text}")
                raise Exception(f"Ollama API error (status {response.status}): {error_text}")
                
            async for line in response.content:
                if line:
                    logger.debug(f"Raw line: {line}")
                    data = json.loads(line)
                    if "response" in data:
                        chunk = data["response"]
                        logger.debug(f"Yielding content: {chunk}")
                        on_chunk(chunk)
                        yield chunk

def create_ollama_client(
    model_name: str = "llama3.2:3b",
    url: str = "http://localhost:11434/api/generate",
    temperature: float = 0.7,
    max_tokens: int = 4096,
    provider: str = "ollama",
    display_name: str = "Local (Ollama LLaMA 3.2)"
) -> llmFn:
    """Create an Ollama LLM client with provided config."""
    
    async def generate_response(
        llm_request: LLMRequest,
        on_chunk: Callable[[str], None]
    ) -> LLMResponse:
        start_time = time.time()
        chunks = []
        
        logger.debug("=== Starting Chunk Processing ===")
        logger.debug(f"Initial chunks array: {chunks}")
        
        try:
            # Get the appropriate prompt based on prompt type
            prompt = (
                llm_request.prompt["selected_agent"] 
                if isinstance(llm_request.prompt, dict) 
                else llm_request.prompt
            )
            
            async for chunk in _stream_ollama_response(
                url,
                model_name,
                prompt,
                on_chunk
            ):
                logger.debug(f"Received chunk type: {type(chunk)}")
                logger.debug(f"Chunk content: {chunk}")
                chunks.append(chunk)
                
            logger.debug("=== Assembling Final Response ===")
            logger.debug(f"All chunks collected: {chunks}")
            response = ''.join(chunks)
            
            raw_response = {"raw_text": response}
            logger.debug(f"Wrapped response type: {type(raw_response)}")
            logger.debug(f"Wrapped response content: {raw_response}")
            
            return LLMResponse(
                generated_at=datetime.datetime.now().isoformat(),
                intents=[],
                request=llm_request,
                raw_response=raw_response,
                model_name=model_name,
                model_provider=provider,
                time_in_seconds=round(time.time() - start_time, 2),
                confidence=0.0
            )
            
        except Exception as e:
            logger.error(f"Error in generate_response: {str(e)}")
            return LLMResponse(
                generated_at=datetime.datetime.now().isoformat(),
                intents=[],
                request=llm_request,
                raw_response={"error": str(e)},
                model_name=model_name,
                model_provider=provider,
                time_in_seconds=0.0,
                confidence=0.0
            )
    
    return generate_response