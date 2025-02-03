import os

from dotenv import load_dotenv

load_dotenv()

def get_ollama_config():
    """Get Ollama LLM configuration."""
    return {
        "model_name": "llama3.2:3b",
        "temperature": 0.7,
        "max_tokens": 4096,
        "provider": "ollama",
        "url": "http://localhost:11434/api/generate",
        "display_name": "Local (Ollama LLaMA 3.2)"
    } 

def get_groq_config():
    """Get Groq LLM configuration."""
    return {
        "model_name": "deepseek-r1-distill-llama-70b",
        "temperature": 0.7,
        "max_tokens": 1024,
        "provider": "groq",
        "api_key": os.getenv("GROQ_API_KEY"),
        "display_name": "Groq (DeepSeek R1 Distill LLaMA 70B)"
    }
