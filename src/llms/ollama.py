import datetime
import time
import json
import aiohttp
from typing import AsyncIterator
from src.data_model import LLMRequest, LLMResponse, OnTextFn, llmFn
from utils.config import get_ollama_config

async def _stream_ollama_response(
    url: str,
    model_name: str,
    prompt: str,
    on_chunk: OnTextFn
) -> AsyncIterator[str]:
    """Stream response from Ollama API."""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            url,
            json={"model": model_name, "prompt": prompt, "stream": True}
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Ollama API error (status {response.status}): {error_text}")
                
            async for line in response.content:
                if line:
                    data = json.loads(line)
                    if "response" in data:
                        chunk = data["response"]
                        on_chunk(chunk)
                        yield chunk

def create_ollama_client() -> llmFn:
    """Create an Ollama LLM client."""
    config = get_ollama_config()
    
    async def generate_response(
        llm_request: LLMRequest,
        on_chunk: OnTextFn
    ) -> LLMResponse:
        start_time = time.time()
        chunks = []
        
        try:
            # Get the appropriate prompt based on intent
            prompt = (
                llm_request.prompt["selected_agent"] 
                if isinstance(llm_request.prompt, dict) 
                else llm_request.prompt
            )
            
            async for chunk in _stream_ollama_response(
                config["url"],
                config["model_name"],
                prompt,
                on_chunk
            ):
                chunks.append(chunk)
            
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
            intent=llm_request.intent,
            request=llm_request,
            raw_response=response,
            model_name=config["model_name"],
            model_provider=config["provider"],
            time_in_seconds=round(time.time() - start_time, 2)
        )
    
    return generate_response