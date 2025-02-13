import os
import datetime
import time
import logging
from typing import Optional
from server.src.data_model import LLMRequest, LLMResponse, OnTextFn, llmFn, Intent
from server.src.llms.groq import create_groq_client
from server.src.agents.pdf_agent import create_pdf_agent
from server.src.agents.meta_agent import analyze_query
from server.src.agents.web_agent import create_web_agent
from server.src.agents.finance_agent import create_finance_agent
from server.src.prompts.prompts import (
    META_AGENT_PROMPT,
    WEB_AGENT_PROMPT,
    PDF_AGENT_PROMPT,
    FINANCE_AGENT_PROMPT
)

logger = logging.getLogger(__name__)

def create_groq_llm() -> llmFn:
    """Factory function to create a Groq LLM client with our prompts."""
    
    logger.debug("Creating Groq LLM client")
    
    # At the top where we create the client, store the values
    model_name = os.getenv("GROQ_MODEL_NAME", "deepseek-r1-distill-llama-70b")
    provider = "groq"
    
    generate_llm_response = create_groq_client(
        model_name=model_name,
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.7,
        max_tokens=4096,
        provider=provider,
        display_name="Groq (DeepSeek R1 Distill LLaMA 70B)"
    )

    async def complete_prompt(
        llm_request: LLMRequest,
        on_chunk: Optional[OnTextFn] = None
    ) -> LLMResponse:
        """Process request through Groq with appropriate prompt."""
        try:
            start_time = time.time()
            
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
            
            # Get contexts based on intents
            pdf_context = None
            web_context = None
            finance_context = None
            
            if Intent.PDF_AGENT in intents:
                pdf_agent = create_pdf_agent()
                pdf_context = await pdf_agent(llm_request.query)
            
            if Intent.WEB_AGENT in intents:
                web_agent = await create_web_agent()
                web_context = await web_agent(llm_request.query)
                
            if Intent.FINANCE_AGENT in intents:
                try:
                    finance_agent = create_finance_agent()
                    finance_context = finance_agent(llm_request.query)
                except Exception as finance_error:
                    logger.error(f"Finance agent error: {finance_error}")
                    finance_context = None
            
            # Build agent prompts
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
                    market_data=finance_context.stock_data if finance_context else "",
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
                confidence=0.8,
                model="groq",
                pdf_context=pdf_context,
                web_context=web_context,
                finance_context=finance_context
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
                model_name=model_name,
                model_provider=provider,
                time_in_seconds=0.0,
                intents=intents if 'intents' in locals() else [Intent.WEB_AGENT],
                confidence=0.0,
                model="groq",
                pdf_context=None,
                web_context=None,
                finance_context=None
            )

    return complete_prompt
