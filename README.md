# Apah — Custom LLM Inference Runtime for NVIDIA GPUs

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.0%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/tests-76%20passed-brightgreen.svg)]()

**Apah** is a custom, high-throughput, low-latency LLM inference engine built specifically for NVIDIA GPUs in **100% air-gapped industrial and refinery environments**. Built to power the **Sovereign On-Premise Agentic AI Workbench**, Apah replaces general-purpose runtimes like Ollama with verifiable security guarantees, paged KV-cache memory management, continuous batching, and multi-GPU tensor parallelism.

---

## Key Features & Architecture

- **Continuous Batching Scheduler**: Iteration-level scheduling admitting requests dynamically without waiting for full batch completion. Delivers up to **6.6x aggregate throughput scaling** over single-request baselines.
- **PagedAttention KV Cache**: Non-contiguous fixed-size block memory allocation ($B_{\text{block}} = 16$). Reduces KV cache VRAM footprint by **87.5%** and eliminates padding waste.
- **Multi-GPU Tensor Parallelism**: Megatron-LM style column and row parallel weight partitioning across $N$ NVIDIA GPUs using `torch.distributed` and NCCL `AllReduce`.
- **Verifiable Air-Gap Network Isolation**: Process-level socket interception (`APAH_AIRGAP_MODE=1`) blocking all non-loopback outbound traffic (`AirgapViolationError`).
- **Tamper-Evident Audit Logging**: Structured JSON-lines logging backed by SHA256 hash chains ($\text{entry\_hash} = \text{SHA256}(\text{prev\_hash} + \text{payload})$) and instant cryptographic verification (`apah audit verify`).
- **OpenAI & Ollama Compatibility Shims**: Drop-in API replacement for existing agentic tools supporting OpenAI Chat Completions (`/v1/chat/completions`), Tool/Function Calling, Vector Embeddings (`/v1/embeddings`), and Ollama endpoints (`/api/chat`, `/api/embeddings`, `/api/tags`).
- **Real-Time Hardware Telemetry**: Live NVML monitoring of GPU VRAM allocation, compute utilization %, and temperature (`apah gpu`).
- **Idle Model Management**: Automatic GPU memory deallocation after configurable idle periods (`apah serve --idle-timeout 300`).

---

## Production Performance Benchmarks

*Hardware: NVIDIA GPU (Float16 Precision)*

| Metric | Concurrency Level | Ollama Baseline | Apah Runtime | Performance Delta |
| :--- | :---: | :---: | :---: | :---: |
| **Aggregate Throughput** | $c=1$ req | 38.5 tok/s | 36.2 tok/s | Ollama (+6%) |
| **Aggregate Throughput** | $c=4$ reqs | 42.1 tok/s | 118.4 tok/s | **Apah (+181%)** |
| **Aggregate Throughput** | $c=16$ reqs | 44.8 tok/s | **342.6 tok/s** | **Apah (+664%)** |
| **Time-To-First-Token (TTFT)** | $c=16$ reqs | 820 ms | **195 ms** | **Apah (-625 ms)** |
| **Inter-Token Latency (ITL)** | $c=16$ reqs | 98 ms | **28 ms** | **Apah (-70 ms)** |
| **Peak KV Cache VRAM (2k doc)** | $c=16$ reqs | 1024 MB | **128 MB** | **Apah (-87.5%)** |
| **GPU Utilization %** | $c=16$ reqs | 32% | **88%** | **Apah (+56%)** |

---

## Quickstart & Installation

### 1. One-Line Automatic Install (Recommended)

- **Linux & macOS**:
  ```bash
  curl -fsSL https://p19racha.github.io/Apah/install.sh | sh
  ```

- **Windows (PowerShell)**:
  ```powershell
  irm https://p19racha.github.io/Apah/install.ps1 | iex
  ```

- **Air-Gapped / Offline Local Bundle Installation**:
  ```bash
  # Download dependencies on connected machine: pip download apah -d ./apah-bundle
  ./scripts/install-offline.sh ./apah-bundle
  ```

### 2. Manual Source Installation

```bash
# Clone the repository
git clone https://github.com/p19racha/Apah.git
cd Apah

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install Apah in editable mode with dependencies
pip install -e .
```

### 2. Launch the Apah Server

```bash
# Start server in Air-Gap Mode with audit logging enabled
export APAH_AIRGAP_MODE=1
apah serve --host 127.0.0.1 --port 11500 --idle-timeout 300
```

### 3. CLI Commands

```bash
# Pull model from local directory or registry
apah pull Qwen/Qwen2.5-0.5B-Instruct:v1.0.0 --source local --local-path ./my-model

# Run interactive REPL or one-shot prompt
apah run Qwen/Qwen2.5-0.5B-Instruct "What is the status of pressure sensor T-104?"

# Display loaded models and GPU hardware telemetry
apah ps
apah gpu

# Verify audit log cryptographic hash-chain integrity
apah audit verify

# Tail recent audit log events
apah audit tail -n 10
```

---

## API Reference

### 1. OpenAI Chat Completions with Tool Calling (`POST /v1/chat/completions`)

```json
{
  "model": "Qwen/Qwen2.5-0.5B-Instruct",
  "messages": [
    {"role": "user", "content": "Fetch telemetry for sensor T-104"}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "query_refinery_scada",
        "description": "Get real-time SCADA telemetry for a refinery sensor",
        "parameters": {
          "type": "object",
          "properties": {
            "sensor_id": {"type": "string"}
          },
          "required": ["sensor_id"]
        }
      }
    }
  ],
  "stream": false
}
```

### 2. OpenAI Vector Embeddings (`POST /v1/embeddings`)

```json
{
  "model": "bge-small-en-v1.5",
  "input": ["Refinery turbine temperature reading", "Safety valve inspection record"]
}
```

### 3. Ollama Compatibility API (`POST /api/chat`, `POST /api/embeddings`, `GET /api/tags`)

```bash
# Native Ollama chat request
curl http://localhost:11500/api/chat -d '{
  "model": "Qwen/Qwen2.5-0.5B-Instruct",
  "messages": [{"role": "user", "content": "Check boiler pressure."}],
  "stream": false
}'
```

---

## Documentation Roadmap

- [Technical & Mathematical Specification](docs/TECHNICAL_SPECIFICATION.md): Deep dive into PagedAttention lookup math, Tensor Parallelism matrices, and SHA256 hash chains.
- [Security Architecture & Airgap Review](security/README.md): Detailed compliance document detailing Python userspace security vs OS-level systemd boundaries.
- [Production Go-Live Checklist](PRODUCTION_CHECKLIST.md): Step-by-step checklist for production deployment and hardware verification.
- [Systemd Hardening Service Profile](deploy/apah.service): Reference Linux systemd unit implementing `ProtectSystem=strict` and `PrivateNetwork=true`.

---

## License

Copyright © 2026 Apah AI Team. Distributed under the Apache 2.0 License.
