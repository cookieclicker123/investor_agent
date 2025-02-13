from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from server.src.web_app import create_web_app
from server.src.ollama_llm import create_ollama_llm
from server.src.groq_llm import create_groq_llm
import os
import logging

logger = logging.getLogger(__name__)

def create_server() -> FastAPI:
    """Create the FastAPI server with all routes configured"""
    
    # Create main FastAPI app
    server = FastAPI(
        title="Investor Agent Server",
        description="Central server for LLM-powered investment analysis",
        version="1.0.0"
    )

    # Add CORS middleware
    server.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Create LLM instances (they now handle their own config)
    ollama_llm = create_ollama_llm()
    groq_llm = create_groq_llm()

    # Create web apps for each LLM
    ollama_app = create_web_app(ollama_llm)
    groq_app = create_web_app(groq_llm)

    # Mount the web apps under their respective paths
    server.mount("/ollama", ollama_app, name="ollama_api")
    server.mount("/groq", groq_app, name="groq_api")

    @server.get("/health")
    async def health_check():
        """Basic health check endpoint"""
        return {
            "status": "healthy",
            "models": {
                "ollama": os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
                "groq": os.getenv("GROQ_MODEL_NAME", "deepseek-r1-distill-llama-70b")
            }
        }

    return server

# Create the server instance
app = create_server()