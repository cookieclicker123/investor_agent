import os
import argparse
import subprocess
import sys
import venv
from pathlib import Path

def create_venv_if_not_exists():
    """Create virtual environment if it doesn't exist"""
    venv_path = Path("tmp/virtual_envs/server")
    if not venv_path.exists():
        print("Creating virtual environment...")
        venv.create(venv_path, with_pip=True)
        
        pip_path = venv_path / "bin" / "pip" if os.name != 'nt' else venv_path / "Scripts" / "pip.exe"
        subprocess.run([str(pip_path), "install", "-r", "server/server_requirements.txt"])
        print("Virtual environment created and requirements installed")
    return venv_path

def create_index(use_langchain: bool = False):
    """Create the index for the PDF agent"""
    venv_path = create_venv_if_not_exists()
    python_path = venv_path / "bin" / "python" if os.name != 'nt' else venv_path / "Scripts" / "python.exe"
    
    base_path = "server/src/langchain_index" if use_langchain else "server/src/index"
    
    print(f"Creating index {'with' if use_langchain else 'without'} langchain...")
    
    # Run pdf_to_json
    result = subprocess.run([
        str(python_path),
        f"{base_path}/pdf_to_json.py"
    ])
    if result.returncode != 0:
        return result.returncode
        
    # Run json_to_index
    result = subprocess.run([
        str(python_path),
        "-m",
        f"{base_path.replace('/', '.')}.json_to_index"
    ])
    return result.returncode

def run_test_suite(python_path: Path, test_path: str):
    """Run a specific test suite"""
    result = subprocess.run([
        str(python_path),
        "-m",
        "pytest",
        test_path,
        "-v"
    ])
    return result.returncode

def run_tests(test_type: str = "all"):
    """Run server tests based on type"""
    venv_path = create_venv_if_not_exists()
    python_path = venv_path / "bin" / "python" if os.name != 'nt' else venv_path / "Scripts" / "python.exe"
    
    test_suites = {
        "data_model": "server/tests/test_data_model.py",
        "llm": "server/tests/test_llm.py",
        "intent": "server/tests/test_intent_extraction.py",
        "prompts": "server/tests/test_prompts.py",
        "groq": "server/tests/test_groq.py",
        "agents": "server/tests/test_llm_agents.py",
        "pdf": "server/tests/test_pdf_agent.py",
        "web": "server/tests/test_web_agent.py",
        "finance": "server/tests/test_finance_agent.py",
        "server": "server/tests/test_server.py",
        "indexing": "server/tests/test_indexing.py",
        "similarity": "server/tests/test_similarity_search.py",
        "all": "server/tests"
    }
    
    if test_type not in test_suites:
        print(f"Unknown test type: {test_type}")
        return 1
        
    print(f"Running {test_type} tests...")
    return run_test_suite(python_path, test_suites[test_type])

def run_terminal_app():
    """Run the terminal-based server app with Groq"""
    venv_path = create_venv_if_not_exists()
    python_path = venv_path / "bin" / "python" if os.name != 'nt' else venv_path / "Scripts" / "python.exe"
    
    # Add root directory to PYTHONPATH
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())  # Add current directory to PYTHONPATH
    env["MODEL"] = "groq"  # Set Groq as default
    env["GROQ_MODEL_NAME"] = "deepseek-r1-distill-llama-70b"  # Specify Groq model
    
    print("Starting terminal app with Groq (DeepSeek R1 Distill LLaMA 70B)...")
    result = subprocess.run([
        str(python_path),
        "server/app.py",
        "--model",
        "deepseek-r1-distill-llama-70b"
    ], env=env)
    return result.returncode

def create_client_venv_if_not_exists():
    """Create virtual environment for client if it doesn't exist"""
    venv_path = Path("tmp/virtual_envs/client")
    if not venv_path.exists():
        print("Creating client virtual environment...")
        venv.create(venv_path, with_pip=True)
        
        pip_path = venv_path / "bin" / "pip" if os.name != 'nt' else venv_path / "Scripts" / "pip.exe"
        subprocess.run([str(pip_path), "install", "-r", "client/client_requirements.txt"])
        print("Client virtual environment created and requirements installed")
    return venv_path

def setup_client():
    """Setup the client environment and show instructions"""
    server_venv = create_venv_if_not_exists()  # Create server venv
    client_venv = create_client_venv_if_not_exists()  # Create client venv
    
    # Get paths for both environments
    server_python = server_venv / "bin" / "python" if os.name != 'nt' else server_venv / "Scripts" / "python.exe"
    server_uvicorn = server_venv / "bin" / "uvicorn" if os.name != 'nt' else server_venv / "Scripts" / "uvicorn.exe"
    client_chainlit = client_venv / "bin" / "chainlit" if os.name != 'nt' else client_venv / "Scripts" / "chainlit.exe"
    
    print("\nBoth environments are ready!")
    print("\nTo run the full application, open two terminal windows and run:")
    print("\nTerminal 1 (Server):")
    print(f"MODEL=groq {server_uvicorn} server.src.server:app --host 0.0.0.0 --port 8006")
    print("\nTerminal 2 (Client):")
    print(f"MODEL=groq {client_chainlit} run client/chainlit/app.py --port 8000")
    print("\nThe application will be available at http://localhost:8000")
    return 0

def test_similarity():
    """Run similarity search test directly"""
    venv_path = create_venv_if_not_exists()
    python_path = venv_path / "bin" / "python" if os.name != 'nt' else venv_path / "Scripts" / "python.exe"
    
    # Add root directory to PYTHONPATH
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())  # Add current directory to PYTHONPATH
    
    print("Running similarity search test...")
    result = subprocess.run([
        str(python_path),
        "server/tests/test_similarity_search.py"
    ], env=env)
    return result.returncode

def main():
    parser = argparse.ArgumentParser(description="Run server in different modes")
    parser.add_argument(
        "mode",
        choices=["index", "test", "terminal", "client", "similarity"],
        help="Mode to run the server in"
    )
    parser.add_argument(
        "--test-type",
        choices=[
            "data_model", "llm", "intent", "prompts", "groq",
            "agents", "pdf", "web", "finance", "server",
            "indexing", "similarity", "all"
        ],
        default="all",
        help="Type of tests to run (default: all)"
    )
    parser.add_argument(
        "--use-langchain",
        action="store_true",
        help="Use langchain for indexing"
    )
    
    args = parser.parse_args()
    
    os.makedirs("tmp/virtual_envs", exist_ok=True)
    
    if args.mode == "index":
        sys.exit(create_index(args.use_langchain))
    elif args.mode == "test":
        sys.exit(run_tests(args.test_type))
    elif args.mode == "terminal":
        sys.exit(run_terminal_app())
    elif args.mode == "client":
        sys.exit(setup_client())
    elif args.mode == "similarity":
        sys.exit(test_similarity())

if __name__ == "__main__":
    main()
