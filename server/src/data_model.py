from pydantic import BaseModel
from enum import Enum
from typing import Dict, List, Any, Callable, Union, Optional
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

class ChunkMetadata(BaseModel):
    """Metadata for document chunks"""
    title: str
    author: str
    creation_date: str
    source_file: str
    chunk_id: int
    total_chunks: int
    chunk_size: int
    chunking_strategy: str

class DocumentChunk(BaseModel):
    """Represents a chunk of text with its metadata"""
    text: str
    metadata: ChunkMetadata
    embedding: Optional[List[float]] = None

class PDFContext(BaseModel):
    """Represents a chunk of context from PDF documents"""
    text: str
    source_file: str
    chunk_id: int
    total_chunks: int
    similarity_score: float

class PDFAgentResponse(BaseModel):
    """Response specific to PDF agent queries"""
    relevant_chunks: List[PDFContext]
    synthesized_answer: Optional[str] = None

class SearchResult(BaseModel):
    """Individual search result from web search"""
    title: str
    snippet: str
    link: str
    date: Optional[str] = None

class WebAgentResponse(BaseModel):
    """Response from web agent including search results"""
    query: str
    search_results: List[SearchResult]
    relevant_results: List[SearchResult]
    generated_at: Optional[str] = None
    error: Optional[str] = None

class StockPrice(BaseModel):
    """Current stock price and trading data"""
    price: float
    change_percent: float
    volume: int
    trading_day: str

class StockFundamentals(BaseModel):
    """Fundamental stock data"""
    market_cap: Optional[str]
    pe_ratio: Optional[str]
    eps: Optional[str]

class StockData(BaseModel):
    """Complete stock information for a symbol"""
    symbol: str
    current_price: StockPrice
    fundamentals: StockFundamentals
    last_updated: str  # ISO format timestamp

class FinanceAgentResponse(BaseModel):
    """Response from finance agent including stock data"""
    query: str
    extracted_symbols: List[str]
    stock_data: List[StockData]
    generated_at: Optional[str] = None
    error: Optional[str] = None

class LLMResponse(BaseModel):
    """Enhanced LLM response to include all agent contexts"""
    generated_at: str
    intents: List[Intent]
    request: LLMRequest
    raw_response: Dict[str, Any]
    model_name: str
    model_provider: str
    time_in_seconds: float
    pdf_context: Optional[PDFAgentResponse] = None
    web_context: Optional[WebAgentResponse] = None
    finance_context: Optional[FinanceAgentResponse] = None
    confidence: float

OnTextFn = Callable[[str], None]

intentFn = Callable[[str], IntentResult]
llmFn = Callable[[str, OnTextFn], LLMResponse]
pdfAgentFn = Callable[[str], PDFAgentResponse]
webAgentFn = Callable[[str], WebAgentResponse]
financeAgentFn = Callable[[str], FinanceAgentResponse]