import datetime
import time
from typing import Dict, Any
import json
import asyncio
from src.data_model import LLMRequest, LLMResponse, OnTextFn, llmFn
from src.intent_extraction import create_intent_detector

def create_mock_llm_client(query_response: Dict[str, Dict[str, Any]], emulation_speed: int = 100) -> llmFn:
    """Creates a mock LLM client that simulates response generation."""
    
    # Create an instance of the intent detector
    detect_intent = create_intent_detector()

    async def generate_llm_response(llm_request: LLMRequest, on_chunk: OnTextFn) -> LLMResponse:
        start_time = time.time()
        
        # Detect intent early
        intent_result = detect_intent(llm_request.query)
        
        response = query_response.get(llm_request.query)
        if response is None:
            raise Exception(f"Query {llm_request.query} not supported")

        # Convert to formatted JSON string
        response_str = json.dumps(response, indent=2)

        if llm_request.as_json:
            response = json.loads(response_str)
            
        # Split response into chunks and stream with delays
        chunk_size = max(1, emulation_speed // 10)
        for i in range(0, len(response_str), chunk_size):
            chunk = response_str[i:i + chunk_size]
            on_chunk(chunk)
            delay = len(chunk) / emulation_speed
            await asyncio.sleep(delay)
        
        return LLMResponse(
            generated_at=datetime.datetime.now().isoformat(),
            intent=intent_result.intent,  # Use the detected intent
            request=llm_request,
            raw_response=response,
            model_name='mock_llm',
            model_provider="mock",
            time_in_seconds=round(time.time() - start_time, 2),
        )
    
    return generate_llm_response
