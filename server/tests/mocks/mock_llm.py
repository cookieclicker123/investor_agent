import datetime
import time
from typing import Dict, Any
import json
import asyncio
from server.src.data_model import LLMRequest, LLMResponse, OnTextFn, llmFn, Intent
from server.src.intent_extraction import create_intent_detector
from server.src.prompts.prompts_for_test import (
    META_AGENT_PROMPT,
    WEB_AGENT_PROMPT,
    PDF_AGENT_PROMPT,
    FINANCE_AGENT_PROMPT
)

def create_mock_llm_client(query_response: Dict[str, Dict[str, Any]], emulation_speed: int = 100) -> llmFn:
    """Creates a mock LLM client that simulates response generation."""
    
    # Create an instance of the intent detector
    detect_intent = create_intent_detector()

    async def generate_llm_response(llm_request: LLMRequest, on_chunk: OnTextFn) -> LLMResponse:
        start_time = time.time()
        
        # Always start with meta agent for intent detection and orchestration
        intent_result = detect_intent(llm_request.query)
        
        # First, format the meta agent prompt with intent information
        meta_prompt = META_AGENT_PROMPT.format(
            meta_history="",     # Would come from conversation history
            available_agents="pdf_agent, web_agent, finance_agent",
            query=llm_request.query,
            detected_intent=intent_result.intent  # Add detected intent to meta prompt
        )
        
        # Based on meta agent's decision (simulated here with intent_result)
        if Intent.PDF_AGENT in intent_result.intent:
            agent_prompt = PDF_AGENT_PROMPT.format(
                pdf_history="",
                context="",
                query=llm_request.query
            )
        elif Intent.WEB_AGENT in intent_result.intent:
            agent_prompt = WEB_AGENT_PROMPT.format(
                web_history="",
                search_results="",
                query=llm_request.query
            )
        elif Intent.FINANCE_AGENT in intent_result.intent:
            agent_prompt = FINANCE_AGENT_PROMPT.format(
                finance_history="",
                market_data="",
                query=llm_request.query
            )
        
        # Store both prompts in the request
        llm_request.prompt = {
            "meta_agent": meta_prompt,
            "selected_agent": agent_prompt
        }
        
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
            intents=intent_result.intent,
            confidence=0.8,
            request=llm_request,
            raw_response=response,
            model_name='mock_llm',
            model_provider="mock",
            time_in_seconds=round(time.time() - start_time, 2)
        )
    
    return generate_llm_response
