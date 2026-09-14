"""FastAPI API routes for OpenAI-compatible chat completions and server management."""

import asyncio
import logging
import time
import uuid
from typing import List, Optional, Union
import torch

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse, StreamingResponse

from apah.api.schemas import (
    ChatCompletionChoice,
    ChatCompletionChoiceMessage,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    CompletionUsage,
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    GPUStatsResponse,
    LoadModelRequest,
    ModelStatus,
    OpenAIError,
    OpenAIErrorResponse,
)
from apah.api.tool_calling import extract_tool_calls, format_tool_instructions
from apah.compat.embeddings import get_embedding_engine
from apah.engine.runtime import ApahRuntime
from apah.engine.sequence import Sequence
from apah.engine.server import server_state

logger = logging.getLogger("apah.api.routes")

router = APIRouter()


@router.post("/v1/embeddings", response_model=EmbeddingResponse, summary="OpenAI-compatible embeddings endpoint")
async def create_embeddings(request: EmbeddingRequest):
    """Generate L2-normalized vector embeddings for input text strings (RAG document workflows)."""
    engine = get_embedding_engine(request.model)
    texts = [request.input] if isinstance(request.input, str) else request.input
    vectors = engine.embed(texts)

    data_list = [
        EmbeddingData(object="embedding", embedding=vec, index=i)
        for i, vec in enumerate(vectors)
    ]
    prompt_tokens = sum(len(t.split()) for t in texts)

    return EmbeddingResponse(
        object="list",
        data=data_list,
        model=request.model,
        usage=CompletionUsage(prompt_tokens=prompt_tokens, completion_tokens=0, total_tokens=prompt_tokens),
    )


def format_chat_prompt(tokenizer, messages: List[ChatMessage], tools: Optional[list] = None) -> str:
    """Format chat messages into a prompt string using tokenizer chat template or ChatML fallback, appending tool instructions if provided."""
    dict_messages = [{"role": m.role, "content": m.content or ""} for m in messages]

    # Prepend tool instructions to system message or first user message if tools are present
    if tools:
        tool_instructions = format_tool_instructions(tools)
        has_system = False
        for msg in dict_messages:
            if msg["role"] == "system":
                msg["content"] = f"{msg['content']}\n\n{tool_instructions}".strip()
                has_system = True
                break
        if not has_system:
            dict_messages.insert(0, {"role": "system", "content": tool_instructions})

    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None) is not None:
        try:
            return tokenizer.apply_chat_template(
                dict_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as e:
            logger.warning(f"Failed to apply tokenizer.chat_template: {e}. Falling back to default format.")

    prompt_parts = []
    for msg in dict_messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            prompt_parts.append(f"<|im_start|>system\n{content}<|im_end|>\n")
        elif role == "user":
            prompt_parts.append(f"<|im_start|>user\n{content}<|im_end|>\n")
        elif role == "assistant":
            prompt_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>\n")
        elif role == "tool":
            prompt_parts.append(f"<|im_start|>tool\n{content}<|im_end|>\n")
    prompt_parts.append("<|im_start|>assistant\n")
    return "".join(prompt_parts)


@router.get("/health", summary="Liveness check endpoint")
async def health():
    """Liveness check endpoint. Does not require a loaded model."""
    return {"status": "ok"}


@router.get("/models", summary="List local model manifests")
async def get_models():
    """List all locally downloaded model manifests in ~/.apah/models/."""
    from apah.registry.manifest import get_default_models_dir, list_versions
    
    models_dir = get_default_models_dir()
    all_manifests = []
    if models_dir.exists():
        for p in models_dir.iterdir():
            if p.is_dir():
                manifests = list_versions(p.name, models_root=models_dir)
                all_manifests.extend(manifests)
    return [m.model_dump() for m in all_manifests]




@router.get("/gpu", response_model=List[GPUStatsResponse], summary="Get real-time NVML hardware stats for visible GPUs")
async def get_gpu_stats():
    """Get real-time hardware telemetry (memory, utilization %, temperature) for visible GPUs."""
    stats_list = server_state.gpu_monitor.get_all_device_stats()
    return [
        GPUStatsResponse(
            index=s.index,
            name=s.name,
            memory_used_mb=s.memory_used_mb,
            memory_total_mb=s.memory_total_mb,
            memory_free_mb=s.memory_free_mb,
            utilization_pct=s.utilization_pct,
            temperature_c=s.temperature_c,
        )
        for s in stats_list
    ]


@router.get("/ps", response_model=List[ModelStatus], summary="Get loaded model and scheduler status")
async def get_process_status():
    """Get status of currently loaded model and continuous batching scheduler."""
    if not server_state.is_loaded or server_state.model_name is None:
        return []

    stats = server_state.scheduler.get_stats()
    tp_size = getattr(server_state.runtime, "tp_world_size", 1)
    idle_sec = server_state.idle_manager.get_idle_seconds()
    gpu_devices = list(range(tp_size))

    return [
        ModelStatus(
            name=server_state.model_name,
            loaded=True,
            gpu_memory_mb=server_state.gpu_memory_mb,
            uptime_seconds=server_state.uptime_seconds,
            idle_seconds=idle_sec,
            active_batch_size=int(stats["active_batch_size"]),
            waiting_queue_length=int(stats["waiting_queue_length"]),
            aggregate_throughput_tok_s=float(stats["aggregate_throughput_tok_s"]),
            tp_world_size=tp_size,
            gpu_devices=gpu_devices,
        )
    ]


@router.post("/load", summary="Load model into runtime")
async def load_model_endpoint(request: LoadModelRequest):
    """Load a Hugging Face model into the Apah runtime and initialize scheduler."""
    if server_state.is_loaded:
        if server_state.model_name == request.model_path:
            return {"status": "success", "message": f"Model '{request.model_path}' is already loaded."}
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={
                "error": f"A model ('{server_state.model_name}') is already loaded. "
                "Unload the existing model first using POST /unload before loading a new one."
            },
        )

    logger.info(f"Received /load request for model='{request.model_path}', dtype='{request.dtype}', tp_world_size={request.tp_world_size}...")

    # Load-time integrity verification for local models
    try:
        from pathlib import Path
        import json
        from apah.registry.manifest import ModelManifest, get_latest, get_model_dir
        from apah.registry.checksum import verify_manifest

        resolved_dir = Path(request.model_path)
        manifest = None
        model_dir = None

        if resolved_dir.exists() and (resolved_dir / "apah_manifest.json").exists():
            model_dir = resolved_dir
            with open(resolved_dir / "apah_manifest.json") as f:
                data = json.load(f)
                if "checksum_per_file" not in data and "checksums" in data:
                    data["checksum_per_file"] = data.pop("checksums")
                manifest = ModelManifest(**data)
        else:
            manifest = get_latest(request.model_path)
            if manifest:
                model_dir = get_model_dir(request.model_path, version=manifest.version)

        if manifest and model_dir and model_dir.exists():
            verification = verify_manifest(model_dir, manifest)
            if not verification.ok:
                logger.error(f"Load-time checksum verification failed for '{request.model_path}': {verification.error_message}")
                server_state.audit_logger.log_event(
                    "auth_failure",
                    details={"error": f"Model integrity check failed for '{request.model_path}'", "reason": verification.error_message},
                )
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={"error": f"Model integrity verification failed: {verification.error_message}. Refusing to load corrupted model into GPU memory."},
                )
    except Exception as check_err:
        logger.warning(f"Could not perform pre-load integrity verification: {check_err}")

    try:
        runtime = ApahRuntime.from_model_path(
            model_path=request.model_path,
            dtype=request.dtype,
            device=request.device,
            tp_world_size=request.tp_world_size,
        )
        server_state.set_runtime(runtime, request.model_path)
        logger.info(f"Model '{request.model_path}' successfully loaded and scheduler initialized.")
        server_state.audit_logger.log_event(
            "model_load",
            details={
                "model_path": request.model_path,
                "dtype": request.dtype,
                "tp_world_size": request.tp_world_size,
            },
        )
        return {"status": "success", "message": f"Model '{request.model_path}' loaded successfully."}
    except Exception as e:
        logger.error(f"Error loading model '{request.model_path}': {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": f"Failed to load model: {str(e)}"},
        )


@router.post("/unload", summary="Unload currently loaded model")
async def unload_model_endpoint():
    """Unload currently loaded model and free GPU memory."""
    if not server_state.is_loaded:
        return {"status": "success", "message": "No model is currently loaded."}

    mem_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0.0
    model_name = server_state.model_name
    server_state.unload()
    mem_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0.0

    freed_mb = (mem_before - mem_after) / (1024 * 1024)
    logger.info(f"Unloaded model '{model_name}'. GPU memory freed: {freed_mb:.2f} MB.")
    server_state.audit_logger.log_event(
        "model_unload",
        details={"model_name": model_name, "freed_memory_mb": round(freed_mb, 2)},
    )
    return {"status": "success", "message": f"Model '{model_name}' unloaded. Freed {freed_mb:.2f} MB GPU memory."}


@router.post("/v1/chat/completions", summary="OpenAI-compatible chat completions endpoint")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint backed by continuous batching scheduler."""
    if not server_state.is_loaded or server_state.runtime is None:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": {
                    "message": "No model is currently loaded. Load a model first using POST /load.",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "model_not_loaded",
                }
            },
        )

    start_time = time.perf_counter()
    req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    server_state.idle_manager.record_activity()
    tokenizer = server_state.runtime.tokenizer
    prompt = format_chat_prompt(tokenizer, request.messages, tools=request.tools)
    prompt_token_ids = tokenizer.encode(prompt)

    stop_token_ids = set()
    if tokenizer.eos_token_id is not None:
        stop_token_ids.add(tokenizer.eos_token_id)

    max_new_tokens = request.max_tokens if request.max_tokens is not None else 256
    temperature = request.temperature if request.temperature is not None else 1.0

    sequence = Sequence(
        prompt=prompt,
        prompt_token_ids=prompt_token_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        stop_token_ids=stop_token_ids,
    )

    # Log audit event: request_received
    req_details = {
        "request_id": req_id,
        "model": request.model or server_state.model_name,
        "stream": request.stream,
        "max_tokens": max_new_tokens,
        "prompt_tokens": len(prompt_token_ids),
        "has_tools": request.tools is not None and len(request.tools) > 0,
    }
    if server_state.audit_logger.log_content:
        req_details["messages"] = [m.model_dump() for m in request.messages]

    server_state.audit_logger.log_event("request_received", details=req_details)

    # Submit request to continuous batching scheduler
    server_state.scheduler.start()
    server_state.scheduler.add_request(sequence)

    logger.info(
        f"Submitted sequence '{sequence.seq_id}' to scheduler: stream={request.stream}, max_tokens={max_new_tokens}"
    )

    if request.stream:
        async def event_generator():
            created_ts = int(time.time())
            model_name = server_state.model_name or request.model
            streamed_chunks = []

            # Initial SSE chunk sending role
            first_chunk = ChatCompletionChunk(
                id=req_id,
                created=created_ts,
                model=model_name,
                choices=[
                    ChatCompletionChunkChoice(
                        index=0,
                        delta=ChatCompletionChunkDelta(role="assistant", content=""),
                        finish_reason=None,
                    )
                ],
            )
            yield f"data: {first_chunk.model_dump_json()}\n\n"

            token_count = 0
            while True:
                chunk_text = await sequence.token_queue.get()
                if chunk_text is None:
                    break

                token_count += 1
                streamed_chunks.append(chunk_text)

                chunk = ChatCompletionChunk(
                    id=req_id,
                    created=created_ts,
                    model=model_name,
                    choices=[
                        ChatCompletionChunkChoice(
                            index=0,
                            delta=ChatCompletionChunkDelta(content=chunk_text),
                            finish_reason=None,
                        )
                    ],
                )
                yield f"data: {chunk.model_dump_json()}\n\n"

            # Final stop chunk
            stop_chunk = ChatCompletionChunk(
                id=req_id,
                created=created_ts,
                model=model_name,
                choices=[
                    ChatCompletionChunkChoice(
                        index=0,
                        delta=ChatCompletionChunkDelta(),
                        finish_reason="stop",
                    )
                ],
            )
            yield f"data: {stop_chunk.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"

            latency = time.perf_counter() - start_time
            logger.info(f"Stream sequence '{sequence.seq_id}' finished: tokens={token_count}, latency={latency:.3f}s")
            
            comp_details = {
                "request_id": req_id,
                "model": model_name,
                "prompt_tokens": len(prompt_token_ids),
                "completion_tokens": token_count,
                "total_tokens": len(prompt_token_ids) + token_count,
                "latency_sec": round(latency, 4),
            }
            if server_state.audit_logger.log_content:
                comp_details["completion_text"] = "".join(streamed_chunks)

            server_state.audit_logger.log_event("request_completed", details=comp_details)

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    else:
        try:
            output_chunks = []
            while True:
                chunk_text = await sequence.token_queue.get()
                if chunk_text is None:
                    break
                output_chunks.append(chunk_text)

            output_text = "".join(output_chunks)
            latency = time.perf_counter() - start_time
            prompt_tokens = len(prompt_token_ids)
            completion_tokens = len(sequence.output_token_ids)
            total_tokens = prompt_tokens + completion_tokens

            cleaned_content, tool_calls = extract_tool_calls(output_text)
            finish_reason = "tool_calls" if tool_calls else "stop"

            logger.info(
                f"Non-streaming sequence '{sequence.seq_id}' finished: "
                f"prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens}, latency={latency:.3f}s, tool_calls={tool_calls is not None}"
            )

            comp_details = {
                "request_id": req_id,
                "model": server_state.model_name or request.model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "latency_sec": round(latency, 4),
            }
            if server_state.audit_logger.log_content:
                comp_details["completion_text"] = output_text

            server_state.audit_logger.log_event("request_completed", details=comp_details)

            return ChatCompletionResponse(
                model=server_state.model_name or request.model,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=ChatCompletionChoiceMessage(
                            role="assistant",
                            content=cleaned_content,
                            tool_calls=tool_calls,
                        ),
                        finish_reason=finish_reason,
                    )
                ],
                usage=CompletionUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                ),
            )
        except Exception as e:
            logger.error(f"Error during sequence '{sequence.seq_id}' generation: {e}", exc_info=True)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": {
                        "message": f"Generation failed: {str(e)}",
                        "type": "internal_error",
                        "param": None,
                        "code": "generation_error",
                    }
                },
            )
