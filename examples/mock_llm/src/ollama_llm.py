import datetime
from typing import Optional
from src.data_model import LLMRequest, LLMResponse, OnTextFn, llmFn, Intent
from src.llms.ollama import create_ollama_client
from src.intent_extraction import create_intent_detector
from src.prompts.prompts import (
    META_AGENT_PROMPT,
    WEB_AGENT_PROMPT,
    PDF_AGENT_PROMPT,
    FINANCE_AGENT_PROMPT
)

def create_ollama_llm() -> llmFn:
    """Factory function to create an Ollama LLM client with our prompts."""
    
    # Set up model configuration
    model_name = 'llama3.2:3b'
    provider = 'ollama'
    
    if provider != "ollama":
        raise ValueError(f"Invalid provider: {provider}")

    # Create the base Ollama client and intent detector
    generate_llm_response = create_ollama_client()
    detect_intent = create_intent_detector()

    async def complete_prompt(
        llm_request: LLMRequest,
        on_chunk: Optional[OnTextFn] = None
    ) -> LLMResponse:
        """Process request through Ollama with appropriate prompt."""
        try:
            # First detect the intent
            intent_result = detect_intent(llm_request.query)
            
            # Format meta agent prompt with detected intent
            meta_prompt = META_AGENT_PROMPT.format(
                meta_history="",
                available_agents="pdf_agent, web_agent, finance_agent",
                query=llm_request.query,
                detected_intent=intent_result.intent
            )

            # Select agent-specific prompt based on detected intent
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
            else:
                raise ValueError(f"Unsupported intent: {intent_result.intent}")

            # Update request with both prompts
            llm_request.prompt = {
                "meta_agent": meta_prompt,
                "selected_agent": agent_prompt
            }

            # Get response from Ollama
            llm_response = await generate_llm_response(
                llm_request=llm_request,
                on_chunk=on_chunk
            )
            
            # Add detected intent to response
            llm_response.intent = intent_result.intent
            
            # Debug: Print the LLM response
            print("LLM Response:", llm_response)

            return llm_response

        except Exception as e:
            print(f"Error in Ollama LLM: {str(e)}")
            return LLMResponse(
                generated_at=datetime.datetime.now().isoformat(),
                intent=intent_result.intent if 'intent_result' in locals() else [],
                request=llm_request,
                raw_response={"error": str(e)},
                model_name=model_name,
                model_provider=provider,
                time_in_seconds=0.0
            )

    return complete_prompt