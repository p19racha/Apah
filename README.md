# Apah — Phase 1: Foundation & Single-Request LLM Inference Runtime

Apah is a custom LLM inference runtime for NVIDIA GPUs, designed to replace Ollama in an air-gapped, on-premise agentic AI workbench.

## Phase 1 Features
- Safetensors format validation (rejecting non-safetensors formats)
- Pre-load GPU memory check before allocation to prevent runtime OOMs
- Model architecture validation (Llama, Mistral, Qwen support)
- Token-by-token generation using direct manual KV caching (`past_key_values`)
- Streaming token generator (`generate_stream`)
- Detailed telemetry: Time-To-First-Token (TTFT), tokens/sec throughput, and peak GPU memory tracking
- Clean memory deallocation on runtime deletion

## Project Structure
```
Apah/
├── pyproject.toml
├── apah/
│   ├── __init__.py
│   └── engine/
│       ├── __init__.py
│       ├── model_loader.py
│       └── runtime.py
└── tests/
    ├── __init__.py
    └── test_runtime.py
```

## Setup & Running Tests

1. Install Apah package in editable mode:
```bash
pip install -e .
```

2. Run Pytest suite:
```bash
pytest tests/test_runtime.py -v -s
```

## Example Usage

```python
from apah.engine import load_model, ApahRuntime

# Load model and tokenizer
model, tokenizer = load_model("Qwen/Qwen2.5-0.5B-Instruct", dtype="bfloat16", device="cuda")

# Initialize Apah runtime
runtime = ApahRuntime(model, tokenizer, device="cuda")

# Non-streaming text generation
output = runtime.generate("Explain quantum computing in one sentence:", max_new_tokens=64)
print(output)

# Streaming text generation
for chunk in runtime.generate_stream("Once upon a time", max_new_tokens=32):
    print(chunk, end="", flush=True)
```
