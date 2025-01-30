# Mock LLM

This is a mock LLM that is used to test the LLM pipeline. It is a simple LLM that returns a mock response, or a real llm with planned responses.

We are testing the robustness of the data model, the intent extraction, and the prompt engineering.

This way, when we move to real llms, we know that the data model and intent extraction are working as expected, without all the moving parts of the real llms, agents, tools, and frontend.

## Run the tests 

```bash
# Move into right path
cd examples/mock_llm

# Run all tests
pytest tests

# Run specific tests
pytest tests/test_mock_llm.py
pytest tests/test_intent_extraction.py
pytest tests/test_prompts.py
pytest tests/test_ollama.py
pytest tests/test_groq.py
```

## Running the mock llm

```bash
python app.py --mock
```

## Running the real llm

```bash
python app.py --model llama3.2:3b
python app.py --model deepseek-r1-distill-llama-70b
```

