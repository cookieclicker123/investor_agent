from typing import Optional
from src.data_model import webAgentFn, WebAgentResponse
from tests.mocks.web_dummy_tools import create_dummy_web_agent

async def create_web_agent(web_search_fn: Optional[webAgentFn] = None) -> webAgentFn:
    """Factory function that creates a web agent using provided or dummy search function"""
    
    # Use provided search function or create dummy one
    search_fn = web_search_fn or create_dummy_web_agent()
    
    async def web_agent(query: str) -> WebAgentResponse:
        """Process web-based queries with search and analysis"""
        try:
            # Get search results using provided function
            return await search_fn(query)
                   
        except Exception as e:
            raise Exception(f"Error in web agent: {str(e)}")
    
    return web_agent 