import chainlit as cl
from typing import Dict, Any
from .streaming import create_streaming_callback

def create_chainlit_stream_handler():
    """Create a streaming handler for Chainlit UI"""
    
    # Closure state
    state = {
        "current_message": None,
        "current_step": None,
        "is_synthesizing": False,
        "text": ""
    }
    
    async def reset_state():
        """Reset handler state between queries"""
        state["current_message"] = None
        state["is_synthesizing"] = False
        state["text"] = ""
        if state["current_step"]:
            await state["current_step"].__aexit__(None, None, None)
            state["current_step"] = None

    async def on_llm_start(metadata: Dict[str, Any]):
        """Handle start of LLM generation"""
        try:
            agent_name = metadata.get('agent_name')
            
            if agent_name:
                if agent_name == "meta":
                    # Close existing step before synthesis
                    if state["current_step"]:
                        await state["current_step"].__aexit__(None, None, None)
                        state["current_step"] = None
                    
                    state["is_synthesizing"] = True
                    await cl.Message(
                        content="# 🧠 Synthesizing Final Analysis...",
                        author="system"
                    ).send()
                else:
                    # Create agent-specific step
                    agent_icons = {
                        "web": "🌐",
                        "finance": "📈",
                        "pdf": "📚"
                    }
                    icon = agent_icons.get(agent_name, "🤖")
                    
                    state["current_step"] = await cl.Step(
                        name=f"{icon} {agent_name.title()} Agent Processing",
                        show_input=False
                    ).__aenter__()
                    
        except Exception as e:
            print(f"Error in on_llm_start: {str(e)}")
            await reset_state()
    
    async def on_llm_token(token: str, **kwargs):
        """Handle streaming tokens"""
        try:
            if state["is_synthesizing"]:
                if state["current_message"] is None:
                    state["current_message"] = await cl.Message(
                        content="",
                        author="Assistant"
                    ).send()
                await state["current_message"].stream_token(token)
            else:
                state["text"] += token
                if state["current_step"]:
                    state["current_step"].output = state["text"]
                    
        except Exception as e:
            print(f"Error streaming token: {str(e)}")
        
    async def on_llm_end(**kwargs):
        """Handle end of LLM generation"""
        try:
            if state["is_synthesizing"]:
                if state["current_message"]:
                    await state["current_message"].update()
                
                await cl.Message(
                    content="# ✨ Analysis Complete!",
                    author="system"
                ).send()
                state["is_synthesizing"] = False
            
            if state["current_step"]:
                await state["current_step"].__aexit__(None, None, None)
                state["current_step"] = None
            
            state["text"] = ""
            
        except Exception as e:
            print(f"Error in on_llm_end: {str(e)}")
            await reset_state()

    async def on_error(error: str):
        """Handle errors during processing"""
        await cl.Message(
            content=f"⚠️ Error: {error}",
            author="system"
        ).send()
        await reset_state()

    return create_streaming_callback(
        on_start=on_llm_start,
        on_token=on_llm_token,
        on_end=on_llm_end,
        on_error=on_error,
        on_reset=reset_state
    )
