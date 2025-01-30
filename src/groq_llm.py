import datetime
import logging
from typing import Optional
from src.data_model import LLMRequest, LLMResponse, OnTextFn, llmFn, Intent
from src.llms.groq import create_groq_client
from src.intent_extraction import create_intent_detector
from utils.config import get_groq_config
from src.prompts.prompts import (
    META_AGENT_PROMPT,
    WEB_AGENT_PROMPT,
    PDF_AGENT_PROMPT,
    FINANCE_AGENT_PROMPT
)

logger = logging.getLogger(__name__)

def create_groq_llm() -> llmFn:
    """Factory function to create a Groq LLM client with our prompts."""
    
    logger.debug("Creating Groq LLM client")
    config = get_groq_config()
    generate_llm_response = create_groq_client()
    detect_intent = create_intent_detector()

    async def complete_prompt(
        llm_request: LLMRequest,
        on_chunk: Optional[OnTextFn] = None
    ) -> LLMResponse:
        """Process request through Groq with appropriate prompt."""
        logger.debug(f"Processing request: {llm_request.query}")
        try:
            # First detect intent
            intent_result = detect_intent(llm_request.query)
            logger.debug(f"Detected intent: {intent_result.intent}")
            
            # Format meta agent prompt
            meta_prompt = META_AGENT_PROMPT.format(
                meta_history="",
                available_agents="pdf_agent, web_agent, finance_agent",
                query=llm_request.query,
                detected_intent=intent_result.intent
            )
            logger.debug("Meta prompt formatted")
            
            # Select agent prompt based on intent
            if Intent.PDF_AGENT in intent_result.intent:
                agent_prompt = PDF_AGENT_PROMPT.format(
                    pdf_history="",
                    context="",
                    query=llm_request.query
                )
                logger.debug("Selected PDF agent")
            elif Intent.WEB_AGENT in intent_result.intent:
                agent_prompt = WEB_AGENT_PROMPT.format(
                    web_history="",
                    search_results="",
                    query=llm_request.query
                )
                logger.debug("Selected Web agent")
            elif Intent.FINANCE_AGENT in intent_result.intent:
                agent_prompt = FINANCE_AGENT_PROMPT.format(
                    finance_history="",
                    market_data="",
                    query=llm_request.query
                )
                logger.debug("Selected Finance agent")
            else:
                logger.error(f"Unsupported intent: {intent_result.intent}")
                raise ValueError(f"Unsupported intent: {intent_result.intent}")
            
            # Update request with prompts
            llm_request.prompt = {
                "meta_agent": meta_prompt,
                "selected_agent": agent_prompt
            }
            logger.debug("Updated request with prompts")

            # Ensure we have a no-op callback if none provided
            if on_chunk is None:
                logger.debug("No chunk callback provided, using no-op")
                on_chunk = lambda x: None

            # Get response from Groq
            logger.debug("Calling Groq client")
            llm_response = await generate_llm_response(
                llm_request=llm_request,
                on_chunk=on_chunk
            )
            logger.debug(f"Got response from Groq: {llm_response.raw_response}")
            
            # Add detected intent to response
            llm_response.intent = intent_result.intent
            
            return llm_response

        except Exception as e:
            logger.error(f"Error in Groq LLM: {str(e)}")
            return LLMResponse(
                generated_at=datetime.datetime.now().isoformat(),
                intent=intent_result.intent if 'intent_result' in locals() else [],
                request=llm_request,
                raw_response={"error": str(e)},
                model_name=config["model_name"],
                model_provider=config["provider"],
                time_in_seconds=0.0
            )

    return complete_prompt
