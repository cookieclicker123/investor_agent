import pytest
import os
import json
from src.data_model import LLMRequest, LLMResponse, Intent
from tests.mocks.mock_llm import create_mock_llm_client

# Ensure the fixtures directory exists
@pytest.fixture(scope="session", autouse=True)
def setup_fixtures_dir():
    fixtures_dir = os.path.join(os.path.dirname(__file__), "fixtures")
    os.makedirs(fixtures_dir, exist_ok=True)
    return fixtures_dir

@pytest.mark.asyncio
async def test_pdf_agent_intent():
    # Mock query response for PDF agent
    query_response = {
        "what are options": {
            "answer": "Options contracts are financial instruments that give the holder the right, but not the obligation, to buy or sell an asset at a predetermined price within a specific time frame"
        }
    }
    
    # Create mock LLM client
    generate_response = create_mock_llm_client(query_response)
    
    # Create request
    request = LLMRequest(query="what are options", prompt="", as_json=True)
    
    # Collect chunks
    chunks = []
    def on_chunk(chunk: str):
        chunks.append(chunk)
    
    # Generate response
    response = await generate_response(request, on_chunk)
    
    # Verify intent
    assert isinstance(response, LLMResponse)
    assert response.intents == [Intent.PDF_AGENT]
    assert response.raw_response == query_response["what are options"]

@pytest.mark.asyncio
async def test_web_agent_intent():
    # Mock query response for Web agent
    query_response = {
        "What impact is deepseek AI's new set of models having on the US stock market this week?": {
            "answer": "it is having a negative impact on tech and AI creator (but not server) stocks due to the threat it poses to the US moat in AI"
        }
    }
    
    # Create mock LLM client
    generate_response = create_mock_llm_client(query_response)
    
    # Create request
    request = LLMRequest(
        query="What impact is deepseek AI's new set of models having on the US stock market this week?",
        prompt="",
        as_json=True
    )
    
    # Collect chunks
    chunks = []
    def on_chunk(chunk: str):
        chunks.append(chunk)
    
    # Generate response
    response = await generate_response(request, on_chunk)
    
    # Verify intent
    assert isinstance(response, LLMResponse)
    assert response.intents == [Intent.WEB_AGENT]
    assert response.raw_response == query_response[request.query]

@pytest.mark.asyncio
async def test_finance_agent_intent():
    # Mock query response for Finance agent
    query_response = {
        "Tell me the performance metrics of PLTR in the stock market this week": {
            "answer": "It has performed very well because palentir will benefit from the open sourcing of models like deepseek-r1 to use as the base model in its own systems due to the open weights nature of the radical and innovative new model"
        }
    }
    
    # Create mock LLM client
    generate_response = create_mock_llm_client(query_response)
    
    # Create request
    request = LLMRequest(
        query="Tell me the performance metrics of PLTR in the stock market this week",
        prompt="",
        as_json=True
    )
    
    # Collect chunks
    chunks = []
    def on_chunk(chunk: str):
        chunks.append(chunk)
    
    # Generate response
    response = await generate_response(request, on_chunk)
    
    # Verify intent
    assert isinstance(response, LLMResponse)
    assert response.intents == [Intent.FINANCE_AGENT]
    assert response.raw_response == query_response[request.query]

    # Log the conversation to fixtures/intent_conversation_log.json
    conversation_log = {
        "request": request.model_dump(),
        "response": response.model_dump(),
        "chunks": chunks
    }
    
    fixtures_path = os.path.join(os.path.dirname(__file__), "fixtures", "intent_conversation_log.json")
    with open(fixtures_path, "w") as log_file:
        json.dump(conversation_log, log_file, indent=2)
