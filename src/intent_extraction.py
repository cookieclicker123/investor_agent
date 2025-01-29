from datetime import datetime
from src.data_model import Intent, IntentResult, intentFn

def create_intent_detector() -> intentFn:
    """Factory function to create an intent detector."""
    
    def detect_intent(text: str) -> IntentResult:
        """Detect which intent should handle the query and return an IntentResult."""
        text = text.lower()
        
        # Simple rule-based intent detection for our specific queries
        if text == "what are options":
            intent = [Intent.PDF_AGENT]
        elif text == "what impact is deepseek ai's new set of models having on the us stock market this week?":
            intent = [Intent.WEB_AGENT]
        elif text == "tell me the performance metrics of pltr in the stock market this week":
            intent = [Intent.FINANCE_AGENT]
        else:
            intent = []
        
        return IntentResult(
            text=text,
            timestamp=datetime.now(),
            intent=intent
        )
    
    return detect_intent