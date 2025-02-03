import pytest
from datetime import datetime
from src.data_model import LLMRequest, LLMResponse, Intent, IntentResult

def test_llm_response_valid_formats():
    base_request = LLMRequest(query="test", prompt="test prompt")
    
    # Dict format should work
    dict_response = LLMResponse(
        generated_at="2024-01-31T15:00:00",
        intents=[Intent.WEB_AGENT],
        confidence=0.8,
        request=base_request,
        raw_response={"raw_text": "response in dict"},
        model_name="test",
        model_provider="test",
        time_in_seconds=1.0
    )
    assert isinstance(dict_response.raw_response, dict)
    assert "raw_text" in dict_response.raw_response

def test_llm_response_streaming_format():
    request = LLMRequest(query="test", prompt="test prompt")
    
    # This simulates our actual streaming response format
    streaming_response = {
        "raw_text": "<think>\nthinking content\n</think>\nactual response"
    }
    
    response = LLMResponse(
        generated_at="2024-01-31T15:00:00",
        intents=[Intent.WEB_AGENT],
        confidence=0.8,
        request=request,
        raw_response=streaming_response,
        model_name="test",
        model_provider="test",
        time_in_seconds=1.0
    )
    assert isinstance(response.raw_response, dict)
    assert "raw_text" in response.raw_response
    assert "<think>" in response.raw_response["raw_text"]

def test_intent_result():
    # Test IntentResult model
    result = IntentResult(
        text="test result",
        timestamp=datetime.now(),
        intent=[Intent.WEB_AGENT]
    )
    assert isinstance(result.text, str)
    assert isinstance(result.intent, list)
    assert result.intent[0] == Intent.WEB_AGENT

def test_llm_request_formats():
    # Test string prompt
    str_request = LLMRequest(
        query="test query",
        prompt="test prompt"
    )
    assert isinstance(str_request.prompt, str)
    
    # Test dict prompt
    dict_request = LLMRequest(
        query="test query",
        prompt={"selected_agent": "test prompt"}
    )
    assert isinstance(dict_request.prompt, dict)
    assert "selected_agent" in dict_request.prompt

def test_invalid_formats():
    request = LLMRequest(query="test", prompt="test prompt")
    
    # Test invalid raw_response type
    with pytest.raises(ValueError):
        LLMResponse(
            generated_at="2024-01-31T15:00:00",
            intents=[Intent.WEB_AGENT],
            confidence=0.8,
            request=request,
            raw_response=123,  # Invalid type
            model_name="test",
            model_provider="test",
            time_in_seconds=1.0
        )
    
    # Test invalid intent type
    with pytest.raises(ValueError):
        LLMResponse(
            generated_at="2024-01-31T15:00:00",
            intents=["invalid_intent"],  # Invalid intent
            confidence=0.8,
            request=request,
            raw_response={"raw_text": "test"},  # Fixed to valid format
            model_name="test",
            model_provider="test",
            time_in_seconds=1.0
        )
