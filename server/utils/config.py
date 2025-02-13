import os

from dotenv import load_dotenv

load_dotenv()


def get_serper_config():
    """Get Serper configuration."""
    return {
        "api_key": os.getenv("SERPER_API_KEY")
    }

def get_alpha_vantage_config():
    """Get Alpha Vantage configuration."""
    return {
        "api_key": os.getenv("ALPHA_VANTAGE_API_KEY")
    }




