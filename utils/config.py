def get_ollama_config():
    """Get Ollama LLM configuration."""
    return {
        "model_name": "llama3.2:3b",
        "temperature": 0.7,
        "max_tokens": 8192,
        "provider": "ollama",
        "url": "http://localhost:11434/api/generate",
        "display_name": "Local (Ollama LLaMA 3.2)"
    } 