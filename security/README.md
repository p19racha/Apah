# Security Architecture & Industrial Compliance Review

**Apah — Sovereign On-Premise Agentic AI Workbench**  
*Phase 9: Verifiable Security Guarantees & Air-Gap Hardening*

---

## 1. Overview & Threat Model

Apah is built specifically for confidential, 100% air-gapped industrial and refinery environments. In these environments:
- Proprietary operational data, process control metrics, and safety document workflows must never leave the on-premise perimeter.
- Outbound network calls (intentional or accidental, including dependency telemetries or Hugging Face automatic model checks) represent a severe compliance violation.
- Log records must be tamper-evident and suitable for strict industrial audit verification.

Apah employs a **Defense-in-Depth Security Model** structured across two distinct layers:
1. **In-Process Python Application Layer** (Apah Runtime Engine)
2. **OS & Kernel Environment Layer** (Systemd, Network Namespaces, Kernel Seccomp)

---

## 2. Security Boundary Matrix

| Security Domain | Application Enforced (Python In-Process) | OS / Environment Enforced (Hard Boundary) |
| :--- | :--- | :--- |
| **Outbound Network Isolation** | `security/network_guard.py` monkey-patches `socket.socket.connect` and `socket.create_connection` to block non-loopback IPs and raise `AirgapViolationError`. | `PrivateNetwork=true` in `systemd` creates an isolated network namespace with loopback only. Outbound packets are dropped by the Linux kernel. |
| **Audit Logging & Integrity** | `security/audit_log.py` generates structured JSON-lines logs with SHA256 hash chains (`entry_hash = SHA256(prev_hash + JSON)`). | Read-only file permissions on historical logs and system log forwarding. |
| **Filesystem Sandboxing** | `security/sandbox.py` validates paths against allowed directories (`~/.apah/models`, `~/.apah/audit_logs`). | `ProtectSystem=strict`, `ProtectHome=true`, `ReadWritePaths=/var/lib/apah/models /var/log/apah/audit` enforced by Linux kernel mount namespaces. |
| **Model Pull Enforcement** | `apah pull --source hf` is hard-disabled at startup when `APAH_AIRGAP_MODE=1`. | Firewall / Air-gap physical network disconnect. |

> [!IMPORTANT]
> **Industrial Compliance Auditor Note**: Python userspace monkey-patching and path validation serve as immediate application defense-in-depth against third-party dependency calls (e.g. `requests`, `httpx`, `huggingface_hub`, `transformers`). However, userspace guards cannot stop native C/C++ extensions or raw `syscall` invocations. **True security boundaries MUST be enforced at the OS level using the provided `deploy/apah.service` systemd unit.**

---

## 3. Network Guard (`security/network_guard.py`)

When `APAH_AIRGAP_MODE=1` is set:
- Low-level socket connections are intercepted.
- Outbound destinations outside `127.0.0.1` / `localhost` trigger an `AirgapViolationError`.
- Server startup outputs a visible banner stating air-gap status.
- CLI model pull requests attempting `--source hf` are rejected instantly.

---

## 4. Tamper-Evident Audit Logging (`security/audit_log.py`)

- **Format**: Structured JSON-lines (`apah_audit_YYYYMMDD.jsonl`).
- **Hash Chain**: Each entry calculates:
  $$\text{entry\_hash} = \text{SHA256}(\text{prev\_hash} + \text{canonical\_json\_payload})$$
- **Log Rotation**: Daily file rotation maintains unbroken hash chain continuity by referencing the last entry hash of the preceding file.
- **Verification**: `apah audit verify` recomputes the entire hash chain from genesis to detect any line modification, insertion, or deletion.
- **Data Privacy**: Default logging captures metadata only (timestamp, model name, prompt/completion token count, latency). Raw prompt/response contents are excluded unless explicitly enabled via `--audit-log-content`.

---

## 5. Deployment Recommendation (Systemd Service)

To deploy Apah in a production refinery environment:
1. Copy `deploy/apah.service` to `/etc/systemd/system/apah.service`.
2. Reload systemd daemon: `systemctl daemon-reload`.
3. Enable and start service: `systemctl enable --now apah.service`.
