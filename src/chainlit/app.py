import sys
from pathlib import Path
import chainlit as cl
from typing import Dict, Any

# Add project root to path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.agents.meta_agent import analyze_query
from src.agents.finance_agent import create_finance_agent
from src.agents.web_agent import create_web_agent
from src.agents.pdf_agent import create_pdf_agent
from src.llms.groq import create_groq_client
from src.chainlit.handlers import create_chainlit_stream_handler
from src.chainlit.ui.components import create_ui_components
from src.chainlit.ui.formatters import create_response_formatters
from src.prompts.prompts import META_AGENT_PROMPT
from src.llms.groq import LLMRequest

# Initialize components
ui = create_ui_components()
formatters = create_response_formatters()

@cl.on_chat_start
async def start():
    """Initialize chat session"""
    # Create streaming handler
    stream_handler = create_chainlit_stream_handler()
    
    # Initialize LLM and agents
    llm = create_groq_client()
    agents = {
        "finance": create_finance_agent(),
        "web": create_web_agent(),
        "pdf": create_pdf_agent()
    }
    
    # Store in session
    cl.user_session.set("llm", llm)
    cl.user_session.set("agents", agents)
    cl.user_session.set("stream_handler", stream_handler)
    
    # Show welcome message
    await cl.Message(
        content="""# 🚀 Financial Expert System

Welcome! This AI-powered system combines multiple specialized agents to provide comprehensive financial analysis:

- 📈 Real-time Market Data
- 🌐 Web Research
- 📚 Educational Resources

Try asking about:
• Stock prices and analysis
• Market trends and news
• Trading strategies and concepts""",
        author="system"
    ).send()

@cl.on_message
async def main(message: cl.Message):
    """Process user messages"""
    try:
        llm = cl.user_session.get("llm")
        agents = cl.user_session.get("agents")
        stream_handler = cl.user_session.get("stream_handler")
        
        # Create LLMRequest object
        llm_request = LLMRequest(
            query=message.content,
            prompt=META_AGENT_PROMPT
        )
        
        # Use the correct function signature with on_chunk handler
        llm_response = await llm(
            llm_request=llm_request,
            on_chunk=stream_handler["on_llm_token"]
        )
        
        # Meta agent analysis
        intents = await analyze_query(llm_response, message.content)
        await ui["show_query_analysis"](llm_response)
        
        # Process through selected agents
        for intent in intents:
            agent_name = intent.value.split("_")[0]
            if agent_name in agents:
                response = await agents[agent_name](message.content)
                await ui[f"show_{agent_name}_analysis"](response)
        
        # Show synthesis header
        await ui["show_synthesis_header"]()
        
    except Exception as e:
        await cl.Message(
            content=formatters["format_error_message"](str(e)),
            author="system"
        ).send()

@cl.on_stop
async def cleanup():
    """Cleanup on chat stop"""
    stream_handler = cl.user_session.get("stream_handler")
    if stream_handler:
        await stream_handler["reset_state"]()
