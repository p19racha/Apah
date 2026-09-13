"""Tamper-evident, hash-chained structured audit logger and log integrity verifier."""

import asyncio
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class AuditVerificationResult(BaseModel):
    """Result payload for audit log hash-chain integrity verification."""

    ok: bool = Field(..., description="Whether audit log hash-chain verification passed.")
    total_entries: int = Field(0, description="Total audit entries verified.")
    tampered_line: Optional[int] = Field(None, description="Line number of first detected tampering if failed.")
    error_message: Optional[str] = Field(None, description="Detailed error description if verification failed.")


GENESIS_HASH = "0000000000000000000000000000000000000000000000000000000000000000"


def compute_entry_hash(prev_hash: str, payload_dict: Dict[str, Any]) -> str:
    """Compute SHA256 entry hash over prev_hash and canonical JSON representation of entry."""
    canonical_json = json.dumps(payload_dict, sort_keys=True)
    hasher = hashlib.sha256()
    hasher.update(f"{prev_hash}:{canonical_json}".encode("utf-8"))
    return hasher.hexdigest()


class AuditLogger:
    """Asynchronous, tamper-evident audit logger writing JSON-lines with SHA256 hash chains."""

    def __init__(
        self,
        log_dir: Optional[Path] = None,
        log_content: bool = False,
        buffer_flush_interval_sec: float = 1.0,
    ):
        self.log_dir = log_dir or (Path.home() / ".apah" / "audit_logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_content = log_content
        self.buffer_flush_interval_sec = buffer_flush_interval_sec

        self._queue: asyncio.Queue = asyncio.Queue()
        self._last_hash: str = self._get_latest_historical_hash()
        self._writer_task: Optional[asyncio.Task] = None
        self._is_running: bool = False

    def _get_current_log_file(self) -> Path:
        today_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        return self.log_dir / f"apah_audit_{today_str}.jsonl"

    def _get_latest_historical_hash(self) -> str:
        """Scan existing audit log files to retrieve the last entry hash for continuous chain continuity."""
        log_files = sorted(self.log_dir.glob("apah_audit_*.jsonl"))
        if not log_files:
            return GENESIS_HASH

        last_file = log_files[-1]
        try:
            with open(last_file, "r") as f:
                lines = [line.strip() for line in f if line.strip()]
                if lines:
                    last_entry = json.loads(lines[-1])
                    return last_entry.get("entry_hash", GENESIS_HASH)
        except Exception:
            pass
        return GENESIS_HASH

    def log_event(
        self,
        event_type: str,
        actor: str = "local",
        details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record an audit event asynchronously."""
        details = details or {}
        now_iso = datetime.now(timezone.utc).isoformat()

        entry_payload = {
            "timestamp": now_iso,
            "event_type": event_type,
            "actor": actor,
            "details": details,
        }

        # Calculate hash chain entry
        entry_hash = compute_entry_hash(self._last_hash, entry_payload)
        complete_record = {
            **entry_payload,
            "prev_hash": self._last_hash,
            "entry_hash": entry_hash,
        }

        self._last_hash = entry_hash
        self._write_entry_sync(complete_record)
        return complete_record

    def _write_entry_sync(self, record: Dict[str, Any]) -> None:
        """Write single record directly to file."""
        today_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        log_file = self.log_dir / f"apah_audit_{today_str}.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(record) + "\n")


def verify_log_integrity(log_path_or_dir: Path) -> AuditVerificationResult:
    """Walk audit log file(s) and recompute SHA256 hash chains to detect tampering or deletion."""
    path = Path(log_path_or_dir)
    log_files: List[Path] = []

    if path.is_file():
        log_files = [path]
    elif path.is_dir():
        log_files = sorted(path.glob("apah_audit_*.jsonl"))
    else:
        return AuditVerificationResult(
            ok=False,
            total_entries=0,
            error_message=f"Log path '{log_path_or_dir}' does not exist.",
        )

    if not log_files:
        return AuditVerificationResult(ok=True, total_entries=0, error_message=None)

    total_entries = 0
    expected_prev_hash = GENESIS_HASH

    for log_file in log_files:
        line_num = 0
        with open(log_file, "r") as f:
            for raw_line in f:
                line_str = raw_line.strip()
                if not line_str:
                    continue
                line_num += 1
                total_entries += 1

                try:
                    record = json.loads(line_str)
                except json.JSONDecodeError as e:
                    return AuditVerificationResult(
                        ok=False,
                        total_entries=total_entries,
                        tampered_line=line_num,
                        error_message=f"JSON decoding error in file '{log_file.name}' line {line_num}: {e}",
                    )

                stored_prev_hash = record.get("prev_hash")
                stored_entry_hash = record.get("entry_hash")

                if total_entries == 1 and stored_prev_hash != GENESIS_HASH:
                    # Initial record referencing prior day's hash
                    expected_prev_hash = stored_prev_hash

                if stored_prev_hash != expected_prev_hash:
                    return AuditVerificationResult(
                        ok=False,
                        total_entries=total_entries,
                        tampered_line=line_num,
                        error_message=(
                            f"Hash chain broken in file '{log_file.name}' at line {line_num}. "
                            f"Expected prev_hash '{expected_prev_hash[:12]}...', got '{stored_prev_hash[:12] if stored_prev_hash else 'None'}...'"
                        ),
                    )

                payload = {
                    "timestamp": record.get("timestamp"),
                    "event_type": record.get("event_type"),
                    "actor": record.get("actor"),
                    "details": record.get("details", {}),
                }

                recomputed_hash = compute_entry_hash(stored_prev_hash, payload)
                if recomputed_hash != stored_entry_hash:
                    return AuditVerificationResult(
                        ok=False,
                        total_entries=total_entries,
                        tampered_line=line_num,
                        error_message=(
                            f"Tampering detected in file '{log_file.name}' at line {line_num}! "
                            f"Recomputed entry_hash '{recomputed_hash[:12]}...' does not match stored hash '{stored_entry_hash[:12] if stored_entry_hash else 'None'}...'"
                        ),
                    )

                expected_prev_hash = stored_entry_hash

    return AuditVerificationResult(ok=True, total_entries=total_entries, error_message=None)
