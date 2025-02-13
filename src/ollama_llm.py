import os
import datetime
import time
import logging
from typing import Optional
from src.data_model import LLMRequest, LLMResponse, OnTextFn, llmFn, Intent
from src.llms.ollama import create_ollama_client
from src.agents.meta_agent import analyze_query
from src.agents.pdf_agent import create_pdf_agent
from src.agents.web_agent import create_web_agent
from src.prompts.prompts import (
    META_AGENT_PROMPT,
    WEB_AGENT_PROMPT,
    PDF_AGENT_PROMPT,
    FINANCE_AGENT_PROMPT
)

logger = logging.getLogger(__name__)

def create_ollama_llm() -> llmFn:
    """Factory function to create an Ollama LLM client with our prompts."""
    
    # At the top where we create the client, store the values
    model_name = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    provider = "ollama"
    
    generate_llm_response = create_ollama_client(
        model_name=model_name,
        url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/api/generate"),
        temperature=0.7,
        max_tokens=4096,
        provider=provider,
        display_name="Local (Ollama LLaMA 3.2)"
    )

    async def complete_prompt(
        llm_request: LLMRequest,
        on_chunk: Optional[OnTextFn] = None
    ) -> LLMResponse:
        """Process request through Ollama with appropriate prompt."""
        start_time = time.time()
        
        try:
            # Meta agent analysis
            meta_request = LLMRequest(
                query=llm_request.query,
                prompt=META_AGENT_PROMPT.format(
                    meta_history="",
                    available_agents="pdf_agent, web_agent, finance_agent",
                    query=llm_request.query,
                    detected_intent=[]
                ),
                as_json=True
            )
            
            meta_response = await generate_llm_response(
                llm_request=meta_request,
                on_chunk=lambda x: None
            )
            
            intents = await analyze_query(
                llm_response=meta_response,
                query=llm_request.query
            )
            
            # Get contexts
            pdf_context = None
            web_context = None
            if Intent.PDF_AGENT in intents:
                pdf_agent = create_pdf_agent()
                pdf_context = await pdf_agent(llm_request.query)

            if Intent.WEB_AGENT in intents:
                web_agent = await create_web_agent()
                web_context = await web_agent(llm_request.query)

            # Build prompts
            agent_prompts = []
            if Intent.PDF_AGENT in intents:
                agent_prompts.append(PDF_AGENT_PROMPT.format(
                    pdf_history="",
                    context=pdf_context.relevant_chunks if pdf_context else "",
                    query=llm_request.query
                ))
            if Intent.WEB_AGENT in intents:
                agent_prompts.append(WEB_AGENT_PROMPT.format(
                    web_history="",
                    search_results=web_context.relevant_results if web_context else "",
                    query=llm_request.query
                ))
            if Intent.FINANCE_AGENT in intents:
                agent_prompts.append(FINANCE_AGENT_PROMPT.format(
                    finance_history="",
                    market_data="",
                    query=llm_request.query
                ))
            
            combined_prompt = "\n\n".join(agent_prompts)
            llm_request.prompt = {
                "meta_agent": meta_response.raw_response,
                "selected_agent": combined_prompt
            }

            if on_chunk is None:
                on_chunk = lambda x: None

            llm_response = await generate_llm_response(
                llm_request=llm_request,
                on_chunk=on_chunk
            )
            
            raw_response = (
                llm_response.raw_response 
                if isinstance(llm_response.raw_response, dict) and 'raw_text' in llm_response.raw_response
                else {"raw_text": str(llm_response.raw_response)}
            )
            
            return LLMResponse(
                generated_at=datetime.datetime.now().isoformat(),
                request=llm_request,
                raw_response=raw_response,
                model_name=model_name,
                model_provider=provider,
                time_in_seconds=time.time() - start_time,
                intents=intents,
                confidence=0.0,
                pdf_context=pdf_context,
                web_context=web_context,
                finance_context=None
            )

        except Exception as e:
            logger.error(f"Error in Ollama LLM: {str(e)}")
            return LLMResponse(
                generated_at=datetime.datetime.now().isoformat(),
                request=llm_request,
                raw_response={
                    "meta_agent": {"raw_text": ""},
                    "raw_text": "",
                    "error": str(e)
                },
                model_name=model_name,
                model_provider=provider,
                time_in_seconds=0.0,
                intents=[Intent.WEB_AGENT],
                confidence=0.0,
                pdf_context=None,
                web_context=None,
                finance_context=None
            )

    return complete_prompt