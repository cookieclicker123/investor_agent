from datetime import datetime
from typing import Callable
from src.data_model import SearchResult, WebAgentResponse, webAgentFn

def create_dummy_web_agent() -> webAgentFn:
    """Factory function that returns a dummy web agent function"""
    
    async def dummy_web_search(query: str) -> WebAgentResponse:
        """Simulated web search that returns dummy results"""
        dummy_results = [
            SearchResult(
                title="Understanding Financial Markets - Investopedia",
                snippet="Comprehensive guide to financial markets, stocks, and trading strategies. Learn about market analysis, technical indicators, and fundamental research.",
                link="https://www.investopedia.com/markets",
                date="2024-02-04"
            ),
            SearchResult(
                title="Latest Market News and Analysis - Financial Times",
                snippet="Breaking news and expert analysis on global markets, companies, and economic trends. Stay informed with our daily market updates.",
                link="https://www.ft.com/markets",
                date="2024-02-04"
            ),
            SearchResult(
                title="Stock Market Data and Research - Yahoo Finance",
                snippet="Real-time stock quotes, financial news, portfolio management resources. Track your investments and market movements.",
                link="https://finance.yahoo.com",
                date="2024-02-04"
            )
        ]
        
        return WebAgentResponse(
            query=query,
            search_results=dummy_results,
            relevant_results=dummy_results[:2],  # Simulate relevance filtering
            generated_at=datetime.now()
        )
    
    return dummy_web_search 