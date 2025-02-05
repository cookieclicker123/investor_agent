#!/bin/bash

# Check if we're running in CLI mode (additional arguments provided)
if [ $# -gt 0 ]; then
    # Check if the command includes deepseek-r1-distill-llama-70b
    if [[ "$*" == *"deepseek-r1-distill-llama-70b"* ]]; then
        echo "Starting CLI with Groq model..."
        export GROQ_MODEL="deepseek-r1-distill-llama-70b"
        MODEL="groq"
    elif [[ "$*" == *"llama3.2:3b"* ]]; then
        echo "Starting CLI with Ollama model..."
        export OLLAMA_MODEL="llama3.2:3b"
        MODEL="ollama"
    fi
    # Get just the arguments after "python app.py"
    shift 2
    exec python app.py "$@"
else
    # API mode - only then set default MODEL
    MODEL=${MODEL:-"ollama"}
    
    case $MODEL in
        "ollama")
            echo "Starting API with Ollama model..."
            python app.py
            ;;
        "groq")
            echo "Starting API with Groq model..."
            python app.py --model deepseek-r1-distill-llama-70b
            ;;
        "mock")
            echo "Starting API with Mock LLM..."
            python app.py --mock
            ;;
        *)
            echo "Invalid model selection. Use: ollama, groq, or mock"
            exit 1
            ;;
    esac
fi 