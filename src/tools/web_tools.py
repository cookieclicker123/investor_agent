from datetime import datetime, timedelta
import requests
from src.data_model import SearchResult, WebAgentResponse, webAgentFn
from utils.config import get_serper_config

def create_web_search() -> webAgentFn:
    """Factory function that creates a web search function using Serper API"""
    
    # Initialize API key and cache
    api_key = get_serper_config().get("api_key")
    if not api_key:
        raise ValueError("SERPER_API_KEY not found in configuration")
        
    base_url = "https://google.serper.dev/search"
    cache = {}
    cache_expiry = {}
    cache_duration = timedelta(minutes=30)
    
    def is_cache_valid(key: str) -> bool:
        if key in cache and key in cache_expiry:
            return datetime.now() < cache_expiry[key]
        return False
    
    async def web_search(query: str) -> WebAgentResponse:
        """Perform web search using Serper API"""
        if not query.strip():  # Handle empty query
            return WebAgentResponse(
                query=query,
                search_results=[],
                relevant_results=[],
                generated_at=datetime.now().isoformat(),
                error="Empty query provided"
            )

        cache_key = query
        
        if is_cache_valid(cache_key):
            return cache[cache_key]
            
        try:
            response = requests.post(
                base_url,
                headers={'X-API-KEY': api_key, 'Content-Type': 'application/json'},
                json={'q': query, 'num': 10}  # Get 10 results for better filtering
            )
            response.raise_for_status()
            results = response.json().get('organic', [])
            
            # Convert to SearchResult objects
            search_results = [
                SearchResult(
                    title=result['title'],
                    snippet=result['snippet'],
                    link=result['link'],
                    date=result.get('date', datetime.now().strftime("%Y-%m-%d"))
                )
                for result in results
            ]
            
            # Create WebAgentResponse
            web_response = WebAgentResponse(
                query=query,
                search_results=search_results,
                relevant_results=search_results[:7],  # Top 7 most relevant
                generated_at=datetime.now().isoformat()  # Set as ISO string
            )
            
            # Cache the response
            cache[cache_key] = web_response
            cache_expiry[cache_key] = datetime.now() + cache_duration
            
            return web_response
            
        except Exception as e:
            return WebAgentResponse(
                query=query,
                search_results=[],
                relevant_results=[],
                generated_at=datetime.now().isoformat(),
                error=str(e)
            )
    
    return web_search 