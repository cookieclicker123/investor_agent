import logging
from typing import List
from server.src.data_model import Intent, LLMResponse

logger = logging.getLogger(__name__)

async def analyze_query(
    llm_response: LLMResponse,
    query: str,
    meta_history: str = ""
) -> List[Intent]:
    """Analyze query using LLM to determine required agents."""
    
    try:
        logger.debug(f"=== META AGENT ANALYSIS ===")
        logger.debug(f"Query: {query}")
        logger.debug(f"Raw response type: {type(llm_response.raw_response)}")
        logger.debug(f"Raw response content: {llm_response.raw_response}")
        
        response_text = llm_response.raw_response
        if isinstance(response_text, dict):
            raw_text = response_text.get("raw_text", "")
            logger.debug(f"Extracted raw_text: {raw_text}")
        else:
            raw_text = response_text
            logger.debug(f"Using response_text directly: {raw_text}")
            
        # Find WORKFLOW section
        workflow_start = raw_text.find("WORKFLOW:")
        logger.debug(f"Workflow start index: {workflow_start}")
        
        if workflow_start != -1:
            workflow_text = raw_text[workflow_start:].split("\n\n")[0].lower()
            logger.debug(f"Found workflow section: {workflow_text}")
            
            # Use a set to prevent duplicates
            intent_set = set()
            
            # Map common names to intents
            agent_mappings = {
                'finance': Intent.FINANCE_AGENT,
                'web': Intent.WEB_AGENT,
                'pdf': Intent.PDF_AGENT,
            }
            
            # Check for agent mentions and add to set
            for agent_name, intent in agent_mappings.items():
                if agent_name in workflow_text:
                    intent_set.add(intent)
                    logger.debug(f"Found agent '{agent_name}' -> {intent}")
            
            if intent_set:
                intents = list(intent_set)
                logger.debug(f"Final deduplicated intents: {intents}")
                return intents
                
        logger.warning(f"No intents detected for query '{query}', defaulting to WEB_AGENT")
        logger.debug("Full response for failed intent detection:")
        logger.debug(raw_text)
        return [Intent.WEB_AGENT]
        
    except Exception as e:
        logger.error(f"Error parsing meta agent response for query '{query}': {e}")
        return [Intent.WEB_AGENT]
