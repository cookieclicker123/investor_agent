from typing import Dict, Any, Callable, Awaitable

def create_streaming_callback(
    on_start: Callable[[Dict[str, Any]], Awaitable[None]] = None,
    on_token: Callable[[str], Awaitable[None]] = None,
    on_end: Callable[[], Awaitable[None]] = None,
    on_error: Callable[[str], Awaitable[None]] = None,
    on_reset: Callable[[], Awaitable[None]] = None
):
    """Create a streaming callback with specified handlers"""
    
    async def default_handler(*args, **kwargs):
        pass
        
    return {
        "on_llm_start": on_start or default_handler,
        "on_llm_token": on_token or default_handler,
        "on_llm_end": on_end or default_handler,
        "on_error": on_error or default_handler,
        "reset_state": on_reset or default_handler
    }

def create_simple_stream_handler():
    """Creates a simple stream handler that prints to stdout"""
    text = ""
    
    async def on_token(token: str, **kwargs):
        nonlocal text
        print(token, end="", flush=True)
        text += token
        
    async def on_end(**kwargs):
        nonlocal text
        text = ""
        
    async def on_error(error: str):
        print(f"\nError: {error}", flush=True)
        
    async def reset_state():
        nonlocal text
        text = ""
        
    return create_streaming_callback(
        on_token=on_token,
        on_end=on_end,
        on_error=on_error,
        on_reset=reset_state
    ) 