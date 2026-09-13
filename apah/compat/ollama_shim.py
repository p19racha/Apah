"""Ollama native API shim for zero-code-change Workbench compatibility."""

import datetime
import json
import time
from typing import Any, Dict, List, Optional, Union
from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from apah.compat.embeddings import get_embedding_engine
from apah.engine.server import server_state
from apah.registry.manifest import list_versions

ollama_router = APIRouter(prefix="/api", tags=["Ollama Compatibility Shim"])


class OllamaChatMessage(BaseModel):
    role: str
    content: str


class OllamaChatRequest(BaseModel):
    model: str
    messages: List[OllamaChatMessage]
    stream: Optional[bool] = True
    options: Optional[Dict[str, Any]] = None


class OllamaGenerateRequest(BaseModel):
    model: str
    prompt: str
    stream: Optional[bool] = True
    options: Optional[Dict[str, Any]] = None


class OllamaEmbeddingRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]


@ollama_router.get("/tags", summary="Ollama list local models endpoint")
async def list_ollama_models():
    """Ollama-compatible GET /api/tags listing local model manifests."""
    from pathlib import Path
    models_dir = Path.home() / ".apah" / "models"
    model_entries = []

    if models_dir.exists():
        for p in models_dir.iterdir():
            if p.is_dir():
                manifests = list_versions(p.name, models_root=models_dir)
                for m in manifests:
                    model_entries.append({
                        "name": f"{m.name}:{m.version}",
                        "model": f"{m.name}:{m.version}",
                        "modified_at": m.pulled_at,
                        "size": m.size_bytes,
                        "digest": m.checksum_sha256,
                        "details": {
                            "format": "safetensors",
                            "family": "llama",
                            "parameter_size": "7B",
                            "quantization_level": m.quant,
                        }
                    })

    if not model_entries and server_state.is_loaded and server_state.model_name:
        model_entries.append({
            "name": server_state.model_name,
            "model": server_state.model_name,
            "modified_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "size": 1000000000,
            "digest": "sha256:loaded",
            "details": {"format": "safetensors", "family": "llama", "parameter_size": "7B", "quantization_level": "none"}
        })

    return {"models": model_entries}


@ollama_router.post("/show", summary="Ollama show model details endpoint")
async def show_ollama_model(request: Dict[str, Any]):
    """Ollama-compatible POST /api/show displaying model details and Modelfile parameters."""
    model_name = request.get("name", "default")
    return {
        "modelfile": f"FROM {model_name}\nPARAMETER temperature 0.7\nPARAMETER top_p 0.9",
        "parameters": "temperature 0.7\ntop_p 0.9",
        "template": "{{ .System }}\nUSER: {{ .Prompt }}\nASSISTANT:",
        "details": {
            "format": "safetensors",
            "family": "llama",
            "parameter_size": "7B",
            "quantization_level": "none",
        }
    }


@ollama_router.post("/embeddings", summary="Ollama native embeddings endpoint")
async def ollama_embeddings(request: OllamaEmbeddingRequest):
    """Ollama-compatible POST /api/embeddings endpoint."""
    engine = get_embedding_engine(request.model)
    embeddings = engine.embed(request.prompt)
    
    if isinstance(request.prompt, str):
        return {"embedding": embeddings[0]}
    return {"embeddings": embeddings}


@ollama_router.post("/chat", summary="Ollama native chat endpoint")
async def ollama_chat(request: OllamaChatRequest):
    """Ollama-compatible POST /api/chat forwarding to Apah chat completion logic."""
    from apah.api.routes import chat_completions
    from apah.api.schemas import ChatCompletionRequest, ChatMessage

    chat_messages = [ChatMessage(role=m.role, content=m.content) for m in request.messages]
    apah_req = ChatCompletionRequest(
        model=request.model,
        messages=chat_messages,
        stream=request.stream,
        temperature=request.options.get("temperature", 1.0) if request.options else 1.0,
    )

    resp = await chat_completions(apah_req)
    if not request.stream:
        if isinstance(resp, JSONResponse):
            return resp
        body = resp.model_dump() if hasattr(resp, "model_dump") else resp
        content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
        return {
            "model": request.model,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "message": {"role": "assistant", "content": content},
            "done": True,
        }
    
    # Streaming shim for Ollama client
    async def ollama_stream_generator():
        # Consume FastAPI StreamingResponse generator lines
        async for chunk in resp.body_iterator:
            line_str = chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk)
            for line in line_str.splitlines():
                if line.startswith("data: ") and not line.endswith("[DONE]"):
                    try:
                        data = json.loads(line[6:])
                        delta_content = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if delta_content:
                            yield json.dumps({
                                "model": request.model,
                                "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                "message": {"role": "assistant", "content": delta_content},
                                "done": False,
                            }) + "\n"
                    except Exception:
                        pass
        yield json.dumps({
            "model": request.model,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "message": {"role": "assistant", "content": ""},
            "done": True,
        }) + "\n"

    return StreamingResponse(ollama_stream_generator(), media_type="application/x-ndjson")


@ollama_router.post("/generate", summary="Ollama native raw generate endpoint")
async def ollama_generate(request: OllamaGenerateRequest):
    """Ollama-compatible POST /api/generate endpoint."""
    from apah.api.routes import chat_completions
    from apah.api.schemas import ChatCompletionRequest, ChatMessage

    chat_messages = [ChatMessage(role="user", content=request.prompt)]
    apah_req = ChatCompletionRequest(
        model=request.model,
        messages=chat_messages,
        stream=request.stream,
        temperature=request.options.get("temperature", 1.0) if request.options else 1.0,
    )

    resp = await chat_completions(apah_req)
    if not request.stream:
        body = resp.model_dump() if hasattr(resp, "model_dump") else resp
        content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
        return {
            "model": request.model,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "response": content,
            "done": True,
        }

    async def ollama_gen_stream_generator():
        async for chunk in resp.body_iterator:
            line_str = chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk)
            for line in line_str.splitlines():
                if line.startswith("data: ") and not line.endswith("[DONE]"):
                    try:
                        data = json.loads(line[6:])
                        delta_content = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if delta_content:
                            yield json.dumps({
                                "model": request.model,
                                "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                "response": delta_content,
                                "done": False,
                            }) + "\n"
                    except Exception:
                        pass
        yield json.dumps({
            "model": request.model,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "response": "",
            "done": True,
        }) + "\n"

    return StreamingResponse(ollama_gen_stream_generator(), media_type="application/x-ndjson")
