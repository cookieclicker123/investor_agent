from pydantic import BaseModel
from enum import Enum
from typing import Dict, List, Any, Callable, Union
from datetime import datetime

class Intent(str, Enum):
    PDF_AGENT = "pdf_agent"
    WEB_AGENT = "web_agent"
    FINANCE_AGENT = "finance_agent"

    @property
    def description(self) -> str:
        return {
            Intent.PDF_AGENT: "Handles PDF Index related queries",
            Intent.WEB_AGENT: "Handles Serper API related queries",
            Intent.FINANCE_AGENT: "Handles Alpha Vantage API related queries",
        }[self]

class IntentResult(BaseModel):
    text: str
    timestamp: datetime
    intent: List[Intent]

class LLMRequest(BaseModel):
    query: str
    prompt: Union[str, Dict[str, str]]
    as_json: bool = False

class LLMResponse(BaseModel):
    generated_at: str
    intent: List[Intent] | None
    request: LLMRequest
    raw_response: str | Dict[str, Any]
    model_name: str
    model_provider: str
    time_in_seconds: float

OnTextFn = Callable[[str], None]

intentFn = Callable[[str], IntentResult]
llmFn = Callable[[LLMRequest, OnTextFn], LLMResponse]