import pytest
from src.data_model import Intent
from src.prompts.prompts import (  # Note: importing from test_prompts
    META_AGENT_PROMPT,
    WEB_AGENT_PROMPT,
    PDF_AGENT_PROMPT,
    FINANCE_AGENT_PROMPT
)

def test_meta_agent_prompt_formatting():
    """Test META_AGENT_PROMPT can be formatted with all required fields."""
    formatted_prompt = META_AGENT_PROMPT.format(
        meta_history="Previous query: What are options?",
        available_agents="pdf_agent, web_agent, finance_agent",
        query="What is the price of (AAPL)?",
        detected_intent=[Intent.FINANCE_AGENT]
    )
    assert "Previous Analysis History" in formatted_prompt
    assert "What is the price of (AAPL)?" in formatted_prompt
    assert "pdf_agent, web_agent, finance_agent" in formatted_prompt
    assert str([Intent.FINANCE_AGENT]) in formatted_prompt  # Verify intent field

def test_web_agent_prompt_formatting():
    """Test WEB_AGENT_PROMPT can be formatted with all required fields."""
    formatted_prompt = WEB_AGENT_PROMPT.format(
        web_history="Previous search: market trends",
        search_results="Latest news about tech stocks...",
        query="What's happening in the market today?"
    )
    assert "Previous Interactions" in formatted_prompt
    assert "What's happening in the market today?" in formatted_prompt
    assert "Latest news about tech stocks..." in formatted_prompt

def test_pdf_agent_prompt_formatting():
    """Test PDF_AGENT_PROMPT can be formatted with all required fields."""
    formatted_prompt = PDF_AGENT_PROMPT.format(
        pdf_history="Previous analysis: options trading basics",
        context="Options are financial derivatives that...",
        query="How do options work?"
    )
    assert "Previous Document Analysis" in formatted_prompt
    assert "How do options work?" in formatted_prompt
    assert "Options are financial derivatives" in formatted_prompt

def test_finance_agent_prompt_formatting():
    """Test FINANCE_AGENT_PROMPT can be formatted with all required fields."""
    formatted_prompt = FINANCE_AGENT_PROMPT.format(
        finance_history="Previous analysis: AAPL Q3 earnings",
        market_data="Current price: 150.23, Volume: 1.2M",
        query="What's the current state of AAPL?"
    )
    assert "Previous Market Analysis" in formatted_prompt
    assert "What's the current state of AAPL?" in formatted_prompt
    assert "Current price: 150.23" in formatted_prompt

def test_prompt_field_validation():
    """Test that prompts fail appropriately when missing required fields."""
    with pytest.raises(KeyError):
        META_AGENT_PROMPT.format(
            meta_history="test",
            # missing available_agents and detected_intent
            query="test"
        )
    
    with pytest.raises(KeyError):
        WEB_AGENT_PROMPT.format(
            web_history="test",
            # missing search_results
            query="test"
        )

def test_prompt_empty_fields():
    """Test prompts work with empty but provided fields."""
    formatted_meta = META_AGENT_PROMPT.format(
        meta_history="",
        available_agents="",
        query="test query",
        detected_intent=""
    )
    assert "test query" in formatted_meta

    formatted_web = WEB_AGENT_PROMPT.format(
        web_history="",
        search_results="",
        query="test query"
    )
    assert "test query" in formatted_web