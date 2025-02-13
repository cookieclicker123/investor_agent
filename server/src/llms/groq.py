import datetime
import time
from typing import AsyncIterator, Callable
import aiohttp
import json
import logging
import ssl
import certifi

from server.src.data_model import LLMRequest, LLMResponse, llmFn

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

def create_groq_client(
    model_name: str,
    api_key: str,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    provider: str = "groq",
    display_name: str = "Groq (DeepSeek R1 Distill LLaMA 70B)"
) -> llmFn:
    """Creates a direct streaming connection to Groq API"""
    
    async def generate_llm_response(
        llm_request: LLMRequest,
        on_chunk: Callable[[str], None]
    ) -> LLMResponse:
        start_time = time.time()
        chunks = []
        
        prompt = (
            llm_request.prompt["selected_agent"] 
            if isinstance(llm_request.prompt, dict) 
            else llm_request.prompt
        )
        
        try:
            async for chunk in make_groq_request(
                model_name=model_name,
                prompt=prompt,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                on_chunk=on_chunk
            ):
                chunks.append(chunk)
            
            response = ''.join(chunks)
            raw_response = {"raw_text": response}
            
            return LLMResponse(
                generated_at=datetime.datetime.now().isoformat(),
                request=llm_request,
                raw_response=raw_response,
                model_name=model_name,
                model_provider=provider,
                time_in_seconds=round(time.time() - start_time, 2),
                confidence=0.0,
                intents=[]
            )
            
        except Exception as e:
            return LLMResponse(
                generated_at=datetime.datetime.now().isoformat(),
                request=llm_request,
                raw_response={"error": str(e)},
                model_name=model_name,
                model_provider=provider,
                time_in_seconds=0.0,
                confidence=0.0,
                intents=[]
            )

    return generate_llm_response