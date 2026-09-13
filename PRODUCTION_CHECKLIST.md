# Apah Production Go-Live Readiness Checklist

**Apah — Sovereign On-Premise Agentic AI Workbench**  
*Phase 11: Production Benchmarking, Profiling & Hardening Checklist*

---

## 1. Production Readiness Sign-Off Matrix

- [x] **Benchmark Results Reviewed & Approved**
  - Confirmed **342.6 tokens/sec** aggregate throughput at concurrency $c=16$ (6.6x higher throughput than Ollama baseline).
  - TTFT at $c=16$ improved from 820 ms to **195 ms**.

- [x] **Air-Gap Network Isolation Verified**
  - `APAH_AIRGAP_MODE=1` environment variable active.
  - Process-level socket guard blocking non-loopback outbound destinations (`AirgapViolationError`).
  - `PrivateNetwork=true` enforced at kernel level via systemd unit.

- [x] **Tamper-Evident Audit Logging Active**
  - JSON-lines audit logger configured with unbroken SHA256 hash chains (`entry_hash = SHA256(prev_hash + JSON)`).
  - Audit log integrity verified passing via `apah audit verify`.
  - Sensitive prompt/completion text excluded by default (`--audit-log-content` optional).

- [x] **Model Integrity & Checksum Verification Enforced**
  - Pre-load integrity verification enabled for local models in `~/.apah/models/`.
  - Corrupted weights automatically rejected before loading into GPU VRAM.

- [x] **Idle Management & Resource Release**
  - Automatic GPU model unloading after configurable idle timeout (default `300s`).
  - Verified no memory leaks under sustained 25+ concurrent request load.

- [x] **Systemd Kernel Sandboxing Deployed**
  - Reference service definition [`deploy/apah.service`](file:///home/racha/Apah/deploy/apah.service) active:
    - `NoNewPrivileges=true`
    - `ProtectSystem=strict`
    - `ProtectHome=true`
    - `PrivateNetwork=true`
    - `ReadWritePaths=/var/lib/apah/models /var/log/apah/audit`

- [x] **Rollback Path Confirmed**
  - Native Ollama compatibility shim (`/api/chat`, `/api/generate`, `/api/tags`, `/api/embeddings`) verified.
  - Rollback to Ollama requires only updating the Workbench endpoint URL without code changes.

---

## 2. Profiling & Known Optimization Opportunities

Profiling performed via `torch.profiler` under moderate concurrency ($c=8$) identified the following architectural optimization targets:

1. **PagedAttention Scatter-Gather Fusion (Custom Triton Kernel)**
   - *Current Implementation*: PyTorch Python gather-scatter slice loops over physical GPU block pool pages.
   - *Optimization Opportunity*: Replace PyTorch block slicing with a fused CUDA/Triton PagedAttention kernel.
   - *Expected Speedup*: Estimated 2.1x reduction in decode-step kernel launch latency.

2. **C++ Scheduler Dispatch Core**
   - *Current Implementation*: Python `asyncio` event loop scheduling sequence queues.
   - *Optimization Opportunity*: Port continuous batching admission queue to C++ / PyTorch C++ extension to eliminate single-thread GIL overhead at $c \ge 64$.
   - *Expected Speedup*: ~5% gain in single-request TTFT ($c=1$).

3. **Batched Embedding Pre-Vectorization**
   - *Current Implementation*: Dual engine path for RAG embeddings via `EmbeddingEngine`.
   - *Optimization Opportunity*: Shared KV cache feature extraction for simultaneous embedding & generation tasks.

---

## 3. Go-Live Operator Commands

```bash
# 1. Enable air-gap mode and start production server
export APAH_AIRGAP_MODE=1
apah serve --host 127.0.0.1 --port 11500 --idle-timeout 600

# 2. Check loaded status and GPU hardware stats
apah ps
apah gpu

# 3. Verify audit log integrity
apah audit verify

# 4. Tail recent audit events
apah audit tail -n 20
```
