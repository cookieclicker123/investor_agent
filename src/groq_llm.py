import datetime
import logging
from typing import Optional, List
from src.data_model import LLMRequest, LLMResponse, OnTextFn, llmFn, Intent
from src.llms.groq import create_groq_client
from tests.mocks.pdf_dummy_tools import get_dummy_context
from src.agents.meta_agent import analyze_query
from utils.config import get_groq_config
from src.prompts.prompts import (
    META_AGENT_PROMPT,
    WEB_AGENT_PROMPT,
    PDF_AGENT_PROMPT,
    FINANCE_AGENT_PROMPT
)
import time
from src.agents.pdf_agent import create_pdf_agent

logger = logging.getLogger(__name__)

def create_groq_llm() -> llmFn:
    """Factory function to create a Groq LLM client with our prompts."""
    
    logger.debug("Creating Groq LLM client")
    config = get_groq_config()
    generate_llm_response = create_groq_client()

    async def complete_prompt(
        llm_request: LLMRequest,
        on_chunk: Optional[OnTextFn] = None
    ) -> LLMResponse:
        """Process request through Groq with appropriate prompt."""
        try:
            start_time = time.time()
            
            # Before creating response
            logger.debug("=== Type Analysis ===")
            logger.debug(f"Request type: {type(llm_request)}")
            logger.debug(f"Request prompt type: {type(llm_request.prompt)}")
            
            # First get meta agent response
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
                on_chunk=lambda x: None  # Silent for meta analysis
            )
            
            # Analyze query to get intents
            intents = await analyze_query(
                llm_response=meta_response,
                query=llm_request.query
            )
            
            # Get PDF context if PDF_AGENT is in intents
            pdf_context = None
            if Intent.PDF_AGENT in intents:
                pdf_agent = create_pdf_agent()
                pdf_context = await pdf_agent(llm_request.query)
            
            # Build agent prompts based on detected intents
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
                    search_results="",
                    query=llm_request.query
                ))
            if Intent.FINANCE_AGENT in intents:
                agent_prompts.append(FINANCE_AGENT_PROMPT.format(
                    finance_history="",
                    market_data="",
                    query=llm_request.query
                ))
            
            # Combine prompts if multiple agents
            combined_prompt = "\n\n".join(agent_prompts)
            
            # Update request with prompts
            llm_request.prompt = {
                "meta_agent": meta_response.raw_response,
                "selected_agent": combined_prompt
            }

            if on_chunk is None:
                on_chunk = lambda x: None

            # Get final response
            llm_response = await generate_llm_response(
                llm_request=llm_request,
                on_chunk=on_chunk
            )
            
            # After getting response from Groq
            logger.debug(f"Raw response from Groq: {type(llm_response.raw_response)}")
            logger.debug(f"Response content structure: {llm_response.raw_response}")
            
            # Ensure response is always a dict with raw_text
            if isinstance(llm_response.raw_response, dict) and 'raw_text' in llm_response.raw_response:
                raw_response = llm_response.raw_response  # Already in correct format
            else:
                raw_response = {"raw_text": str(llm_response.raw_response)}  # Convert to correct format
            
            # Return response with PDF context if present
            return LLMResponse(
                generated_at=datetime.datetime.now().isoformat(),
                request=llm_request,
                raw_response=raw_response,
                model_name=config["model_name"],
                model_provider=config["provider"],
                time_in_seconds=time.time() - start_time,
                intents=intents,
                confidence=0.8,
                model="groq",
                pdf_context=pdf_context  # Add PDF context when present
            )

        except Exception as e:
            logger.error(f"Error in Groq LLM: {str(e)}")
            return LLMResponse(
                generated_at=datetime.datetime.now().isoformat(),
                request=llm_request,
                raw_response={
                    "meta_agent": {"raw_text": ""},
                    "raw_text": "",
                    "error": str(e)
                },
                model_name=config["model_name"],
                model_provider=config["provider"],
                time_in_seconds=0.0,
                intents=[Intent.WEB_AGENT],
                confidence=0.0,
                model="groq",
                pdf_context=None
            )

    return complete_prompt
