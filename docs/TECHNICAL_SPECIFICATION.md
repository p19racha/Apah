# Apah Inference Engine: Technical & Mathematical Specification

**Apah — Custom LLM Inference Runtime for NVIDIA GPUs**  
*Sovereign On-Premise Agentic AI Workbench (Industrial & Refinery Environments)*

---

## 1. Architectural Principles

Apah is a low-latency, high-throughput custom LLM inference engine designed from first principles to run on NVIDIA GPUs in 100% air-gapped confidential environments.

```
                  +-------------------------------------------------------+
                  |    Sovereign Workbench Agentic Applications           |
                  +-------------------------------------------------------+
                                      |
                     +----------------+----------------+
                     |                                 |
         OpenAI-Compatible API                 Ollama Native Shim
     (/v1/chat/completions, /v1/embeddings)   (/api/chat, /api/embeddings)
                     |                                 |
                     +----------------+----------------+
                                      |
                           +----------------------+
                           |  FastAPI Server Layer|
                           +----------------------+
                                      |
              +-----------------------+-----------------------+
              |                       |                       |
   +--------------------+   +-------------------+   +------------------+
   | Network & Airgap   |   | Tamper-Evident    |   | Python Path      |
   | Guard (Socket Patch|   | Audit Logger      |   | Sandbox Guard    |
   +--------------------+   +-------------------+   +------------------+
                                      |
                     +----------------+----------------+
                     | Continuous Batching Scheduler   |
                     +---------------------------------+
                                      |
                     +----------------+----------------+
                     |  PagedAttention Memory Engine   |
                     |  (BlockPool & BlockTable)       |
                     +---------------------------------+
                                      |
                     +----------------+----------------+
                     |  Tensor Parallel Execution      |
                     |  (NCCL AllReduce / Column-Row)  |
                     +---------------------------------+
                                      |
                     +----------------+----------------+
                     |  CUDA Hardware Execution Engine |
                     +---------------------------------+
```

---

## 2. Mathematical Formalisms & Core Algorithms

### 2.1 Autoregressive Decoding & Manual KV-Cache Decoding

Let $X = (x_1, x_2, \dots, x_n)$ represent an input sequence of tokens. The autoregressive probability of generating sequence $X$ is factorized as:

$$P(X) = \prod_{t=1}^n P(x_t \mid x_1, x_2, \dots, x_{t-1})$$

At each generation step $t$, the hidden state vector $h_t \in \mathbb{R}^d$ is computed by the Transformer stack. The logits vector $z_t \in \mathbb{R}^{V}$ over vocabulary size $V$ is given by:

$$z_t = W_{\text{vocab}} \cdot h_t$$

The next-token probability distribution is obtained via temperature-scaled Softmax:

$$P(x_t = k \mid x_{<t}) = \frac{\exp(z_{t,k} / \tau)}{\sum_{j=1}^{V} \exp(z_{t,j} / \tau)}$$

where $\tau > 0$ is the sampling temperature.

To avoid recomputing key-value activations for historical tokens $x_{<t}$ at every step (which would require $O(n^2)$ compute per step), Apah stores key and value tensors in a cache:

$$K_{\le t} = [K_{<t} \parallel K_t], \quad V_{\le t} = [V_{<t} \parallel V_t]$$

The attention computation at step $t$ reduces from $O(t^2)$ to $O(t)$ per layer:

$$\text{Attention}(Q_t, K_{\le t}, V_{\le t}) = \text{Softmax}\left( \frac{Q_t K_{\le t}^T}{\sqrt{d_k}} \right) V_{\le t}$$

---

### 2.2 PagedAttention Memory Management Engine

Standard KV-cache allocations reserve contiguous VRAM tensors padded to the maximum sequence length $L_{\text{max}}$. For batch size $B$, layers $L$, KV heads $H_{kv}$, and head dimension $D$, traditional padded memory allocation requires:

$$\mathcal{M}_{\text{padded}} = 2 \cdot B \cdot L \cdot H_{kv} \cdot L_{\text{max}} \cdot D \cdot b_{\text{bytes}}$$

Because real sequence lengths $L_i \ll L_{\text{max}}$, this results in fragmentation waste $\mathcal{W}$:

$$\mathcal{W} = 1 - \frac{\sum_{i=1}^B L_i}{B \cdot L_{\text{max}}}$$

Apah solves this using **PagedAttention**, partitioning VRAM into fixed-size physical blocks of size $B_{\text{block}} = 16$ tokens.

#### Block Table Lookup Mapping
Let sequence $s$ have prompt length $L_s$. The physical block index $P(s, i)$ for logical token index $i \in [0, L_s - 1]$ is determined by:

$$P(s, i) = \text{BlockTable}[s]\left[ \left\lfloor \frac{i}{B_{\text{block}}} \right\rfloor \right]$$

The offset within the physical block is:

$$O(i) = i \pmod{B_{\text{block}}}$$

Physical Key and Value tensors are retrieved without contiguous allocation:

$$K_{\text{physical}}(s, i) = \text{BlockPool}\left[ P(s, i), \, O(i), \, \text{layer\_idx} \right]$$

#### Paged VRAM Footprint Equation
The actual memory allocated under PagedAttention is strictly proportional to the ceiling number of blocks required:

$$\mathcal{M}_{\text{paged}} = 2 \cdot L \cdot H_{kv} \cdot D \cdot b_{\text{bytes}} \cdot B_{\text{block}} \cdot \sum_{i=1}^B \left\lceil \frac{L_i}{B_{\text{block}}} \right\rceil$$

This guarantees memory waste is bounded by less than $B_{\text{block}}$ tokens per sequence:

$$\mathcal{W}_{\text{paged}} < \frac{B_{\text{block}}}{L_i} \le \frac{16}{128} = 12.5\%$$

---

### 2.3 Multi-GPU Tensor Parallelism (Megatron-LM Style)

Apah splits large model weights across $N$ GPUs using `torch.distributed` with NCCL communication primitives.

#### Column-Parallel Layers (Attention $W_Q, W_K, W_V$ and MLP Gate/Up $W_{\text{gate}}, W_{\text{up}}$)
The weight matrix $W \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$ is split along columns into $N$ equal shards:

$$W = \begin{bmatrix} W_1 & W_2 & \dots & W_N \end{bmatrix}, \quad W_i \in \mathbb{R}^{d_{\text{in}} \times \frac{d_{\text{out}}}{N}}$$

Each GPU $i$ independently computes its local output without inter-GPU communication:

$$Y_i = X \cdot W_i$$

#### Row-Parallel Layers (Attention Output $W_O$ and MLP Down $W_{\text{down}}$)
The weight matrix $W \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$ is split along rows into $N$ equal shards:

$$W = \begin{bmatrix} W_1 \\ W_2 \\ \vdots \\ W_N \end{bmatrix}, \quad W_i \in \mathbb{R}^{\frac{d_{\text{in}}}{N} \times d_{\text{out}}}$$

Input $X$ is sharded column-wise across GPUs as $X_i \in \mathbb{R}^{b \times \frac{d_{\text{in}}}{N}}$. Each GPU computes local product $X_i W_i$, followed by an NCCL `AllReduce-Sum` to combine results across ranks:

$$Y = \text{AllReduce-Sum}\left( \sum_{i=1}^N X_i W_i \right) = \sum_{i=1}^N X_i W_i$$

```
Rank 0: X_0 ---> [ W_0 ] ---> Y_0 --+
                                    |---> [ NCCL AllReduce-Sum ] ---> Y (Final Output)
Rank 1: X_1 ---> [ W_1 ] ---> Y_1 --+
```

---

### 2.4 Tamper-Evident SHA256 Hash-Chained Audit Logging

To guarantee industrial compliance review capabilities, Apah logs every administrative, model loading, and inference request event in a tamper-evident hash chain format.

Let $E_k$ represent the $k$-th structured JSON event record. Let $H_k \in \mathbb{H}^{256}$ be the 256-bit hexadecimal SHA256 digest of record $k$.

#### Genesis Entry ($k = 0$)
$$H_0 = \text{"0000000000000000000000000000000000000000000000000000000000000000"}$$

#### Hash Chain Entry Computation ($k \ge 1$)
$$H_k = \text{SHA256}\Big( H_{k-1} \parallel \text{":"} \parallel \text{CanonicalJSON}(E_k) \Big)$$

where $\text{CanonicalJSON}(E_k)$ converts $E_k$ into a deterministic string with lexicographically sorted dictionary keys.

#### Verification Condition
The audit verification function $\mathcal{V}(\text{log\_file})$ recomputes expected hashes $H_k'$ from line $1$ to $N$:

$$\mathcal{V} = \bigwedge_{k=1}^N \Big( H_k' == H_k \quad \text{and} \quad H_{k-1}' == \text{prev\_hash}_k \Big)$$

If any historic record $E_m$ is modified, inserted, or deleted ($m \le N$), all downstream hashes $H_j$ ($j \ge m$) break, immediately identifying the exact line index of tampering.

---

## 3. Process Isolation & Airgap Network Isolation

When `APAH_AIRGAP_MODE=1` is set:
1. **Low-Level Socket Patching**: Monkey-patches `socket.socket.connect` and `socket.create_connection`. Non-loopback outbound destinations raise `AirgapViolationError`.
2. **Loopback Exception**: `127.0.0.1`, `localhost`, `::1` connections remain enabled for internal local API communication.
3. **OS Kernel Boundary**: Deploys `PrivateNetwork=true`, `ProtectSystem=strict`, `ProtectHome=true`, `NoNewPrivileges=true` via systemd (`deploy/apah.service`).

---

## 4. Benchmark & Performance Methodology

- **Throughput Metrics**:
  - Prefill Throughput ($\text{Tok/s}_{\text{prefill}}$) = $\frac{\sum \text{Prompt Tokens}}{\text{Prefill Time}}$
  - Decode Throughput ($\text{Tok/s}_{\text{decode}}$) = $\frac{\sum \text{Generated Tokens}}{\text{Decode Time}}$
  - Aggregate Throughput ($\text{Tok/s}_{\text{total}}$) = $\frac{\text{Total Tokens}}{\text{Wall Time}}$
- **Latency Percentiles**:
  - Time-To-First-Token ($\text{TTFT}$): $t_{\text{token}_1} - t_{\text{request\_submit}}$
  - Inter-Token Latency ($\text{ITL}$): $t_{\text{token}_k} - t_{\text{token}_{k-1}}$
