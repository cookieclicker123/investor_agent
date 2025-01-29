import pytest
import os
import json
from src.data_model import LLMRequest, LLMResponse
from src.mock_llm import create_mock_llm_client

# Ensure the fixtures directory exists
@pytest.fixture(scope="session", autouse=True)
def setup_fixtures_dir():
    fixtures_dir = os.path.join(os.path.dirname(__file__), "fixtures")
    os.makedirs(fixtures_dir, exist_ok=True)
    return fixtures_dir

@pytest.mark.asyncio
async def test_generate_llm_response_structure():
    # Mock query response
    query_response = {
        "What is AI?": {"answer": "AI is the simulation of human intelligence in machines."}
    }
    
    # Create a mock LLM client
    generate_response = create_mock_llm_client(query_response)
    
    # Create a fake LLM request
    request = LLMRequest(query="What is AI?", prompt="", as_json=True)
    
    # Define a simple on_chunk function
    chunks = []
    def on_chunk(chunk: str):
        chunks.append(chunk)
    
    # Generate a response
    response = await generate_response(request, on_chunk)
    
    # Assert the response is an instance of LLMResponse
    assert isinstance(response, LLMResponse)
    
    # Assert the response fields
    assert response.raw_response == query_response["What is AI?"]
    assert response.model_name == "mock_llm"
    assert response.model_provider == "mock"
    assert response.request == request


@pytest.mark.asyncio
async def test_generate_llm_response_chunks():
    # Mock query response
    query_response = {
        "What is AI?": {"answer": "AI is the simulation of human intelligence in machines."}
    }
    
    # Create a mock LLM client
    generate_response = create_mock_llm_client(query_response)
    
    # Create a fake LLM request
    request = LLMRequest(query="What is AI?", prompt="", as_json=True)
    
    # Define a simple on_chunk function
    chunks = []
    def on_chunk(chunk: str):
        chunks.append(chunk)
    
    # Generate a response
    await generate_response(request, on_chunk)
    
    # Assert that chunks were received
    assert len(chunks) > 0
    # Optionally, check the content of the chunks
    full_response = ''.join(chunks)
    assert json.loads(full_response) == query_response["What is AI?"]


@pytest.mark.asyncio
async def test_conversation_log():
    # Mock query response
    query_response = {
        "What is AI?": {"answer": "AI is the simulation of human intelligence in machines."}
    }
    
    # Create a mock LLM client
    generate_response = create_mock_llm_client(query_response)
    
    # Create a fake LLM request
    request = LLMRequest(query="What is AI?", prompt="", as_json=True)
    
    # Define a simple on_chunk function
    chunks = []
    def on_chunk(chunk: str):
        chunks.append(chunk)
    
    # Generate a response
    response = await generate_response(request, on_chunk)
    
    # Log the conversation to a JSON file
    conversation_log = {
        "request": request.model_dump(),
        "response": response.model_dump(),
        "chunks": chunks
    }
    
    fixtures_path = os.path.join(os.path.dirname(__file__), "fixtures", "mock_llm_conversation_log.json")
    with open(fixtures_path, "w") as log_file:
        json.dump(conversation_log, log_file, indent=2)

    # Assert the log file was created and contains the expected data
    with open(fixtures_path, "r") as log_file:
        logged_data = json.load(log_file)
        assert logged_data["request"] == request.model_dump()
        assert logged_data["response"] == response.model_dump()
