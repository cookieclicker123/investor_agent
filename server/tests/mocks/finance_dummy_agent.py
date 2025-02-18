from datetime import datetime
from server.src.data_model import FinanceAgentResponse, financeAgentFn
from server.tests.mocks.finance_dummy_tools import dummy_finance_search

def create_finance_agent(use_dummy: bool = True) -> financeAgentFn:
    """
    Factory function that returns a finance agent function
    
    Args:
        use_dummy (bool): Whether to use dummy data (default: True)
        
    Returns:
        financeAgentFn: Function that processes finance queries
    """
    
    def process_finance_query(query: str) -> FinanceAgentResponse:
        """
        Process financial queries and return structured stock data
        
        Args:
            query (str): User query containing stock symbols
            
        Returns:
            FinanceAgentResponse: Structured response with stock data
        """
        try:
            # Use dummy search for now, we'll add real search later
            search_fn = dummy_finance_search if use_dummy else None
            
            if not search_fn:
                raise ValueError("No finance search function configured")
                
            # Get financial data using configured search function
            response = search_fn(query)
            
            # Ensure generated_at is set
            if not response.generated_at:
                response.generated_at = datetime.now().isoformat()
                
            return response
            
        except Exception as e:
            # Return error response in expected format
            return FinanceAgentResponse(
                query=query,
                extracted_symbols=[],
                stock_data=[],
                generated_at=datetime.now().isoformat(),
                error=str(e)
            )
            
    return process_finance_query