from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.web_app import create_web_app
from src.ollama_llm import create_ollama_llm
from src.groq_llm import create_groq_llm
from utils.config import get_ollama_config, get_groq_config
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

    # Initialize LLMs based on config
    ollama_config = get_ollama_config()
    groq_config = get_groq_config()

    # Create LLM instances
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
                "ollama": ollama_config["model_name"],
                "groq": groq_config["model_name"]
            }
        }

    return server

# Create the server instance
app = create_server()