from typing import Optional
from datetime import datetime
from src.data_model import FinanceAgentResponse, financeAgentFn
from src.tools.finance_tools import finance_search
from tests.mocks.finance_dummy_tools import dummy_finance_search

def create_finance_agent(use_dummy: bool = True) -> financeAgentFn:
    """
    Factory function that returns a finance agent function
    
    Args:
        use_dummy (bool): Whether to use dummy data (default: True)
        
    Returns:
        financeAgentFn: Function that processes finance queries
    """
    
    # Select the appropriate search function
    search_fn = dummy_finance_search if use_dummy else finance_search
    
    def process_finance_query(query: str) -> FinanceAgentResponse:
        """
        Process financial queries and return structured stock data
        
        Args:
            query (str): User query containing stock symbols
            
        Returns:
            FinanceAgentResponse: Structured response with stock data
        """
        try:
            # Always include fundamentals for complete analysis
            response = search_fn(query, include_fundamentals=True)
            
            # Ensure generated_at is set
            if not response.generated_at:
                response.generated_at = datetime.now().isoformat()
                
            return response
            
        except Exception as e:
            return FinanceAgentResponse(
                query=query,
                extracted_symbols=[],
                stock_data=[],
                generated_at=datetime.now().isoformat(),
                error=str(e)
            )
            
    return process_finance_query 