from datetime import datetime
from server.src.data_model import Intent, IntentResult, intentFn

def create_intent_detector() -> intentFn:
    """Factory function to create an intent detector."""
    
    def detect_intent(text: str) -> IntentResult:
        """Detect which intent should handle the query and return an IntentResult."""
        text = text.lower()
        
        # Rule-based intent detection for financial queries
        if '(aapl)' in text or 'pltr' in text or any(f'({ticker})' in text for ticker in ['msft', 'googl', 'tsla']):
            intent = [Intent.FINANCE_AGENT]
        elif 'options' in text or 'trading' in text or 'investment' in text:
            intent = [Intent.PDF_AGENT]
        elif 'market' in text or 'news' in text or "what's happening" in text:
            intent = [Intent.WEB_AGENT]
        else:
            intent = []
        
        return IntentResult(
            text=text,
            timestamp=datetime.now(),
            intent=intent
        )
    
    return detect_intent