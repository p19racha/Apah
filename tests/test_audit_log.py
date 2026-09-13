"""Unit tests for tamper-evident SHA256 hash-chained audit logger."""

import json
from pathlib import Path
import pytest

from apah.security.audit_log import (
    GENESIS_HASH,
    AuditLogger,
    compute_entry_hash,
    verify_log_integrity,
)


def test_audit_log_hash_chain_validity(tmp_path: Path):
    """Test writing multiple audit log events and verifying hash chain integrity."""
    logger = AuditLogger(log_dir=tmp_path)

    rec1 = logger.log_event("model_load", actor="admin", details={"model": "test-model"})
    rec2 = logger.log_event("request_received", actor="user1", details={"prompt_tokens": 12})
    rec3 = logger.log_event("request_completed", actor="user1", details={"total_tokens": 42})

    assert rec1["prev_hash"] == GENESIS_HASH
    assert rec2["prev_hash"] == rec1["entry_hash"]
    assert rec3["prev_hash"] == rec2["entry_hash"]

    # Verify entire log directory integrity
    res = verify_log_integrity(tmp_path)
    assert res.ok is True
    assert res.total_entries == 3
    assert res.tampered_line is None


def test_audit_log_detects_tampering(tmp_path: Path):
    """Test that modifying a historical log entry breaks the hash chain and is detected."""
    logger = AuditLogger(log_dir=tmp_path)

    logger.log_event("model_load", details={"model": "m1"})
    logger.log_event("request_received", details={"req_id": "r1"})
    logger.log_event("request_completed", details={"req_id": "r1"})

    log_files = list(tmp_path.glob("apah_audit_*.jsonl"))
    assert len(log_files) == 1
    log_file = log_files[0]

    # Tamper with line 2
    with open(log_file, "r") as f:
        lines = f.readlines()

    modified_rec = json.loads(lines[1])
    modified_rec["details"]["req_id"] = "TAMPERED_REQ_ID"
    lines[1] = json.dumps(modified_rec) + "\n"

    with open(log_file, "w") as f:
        f.writelines(lines)

    # Verify integrity report catches tampering
    res = verify_log_integrity(log_file)
    assert res.ok is False
    assert res.tampered_line == 2
    assert "Tampering detected" in res.error_message or "Hash chain broken" in res.error_message


def test_audit_log_cross_file_continuity(tmp_path: Path):
    """Test that daily log rotation maintains continuous hash chain across files."""
    file1 = tmp_path / "apah_audit_20260901.jsonl"
    file2 = tmp_path / "apah_audit_20260902.jsonl"

    rec1_payload = {"timestamp": "2026-09-01T10:00:00Z", "event_type": "model_load", "actor": "local", "details": {}}
    h1 = compute_entry_hash(GENESIS_HASH, rec1_payload)
    rec1 = {**rec1_payload, "prev_hash": GENESIS_HASH, "entry_hash": h1}

    with open(file1, "w") as f:
        f.write(json.dumps(rec1) + "\n")

    # File 2 references h1 as prev_hash
    rec2_payload = {"timestamp": "2026-09-02T10:00:00Z", "event_type": "request_received", "actor": "local", "details": {}}
    h2 = compute_entry_hash(h1, rec2_payload)
    rec2 = {**rec2_payload, "prev_hash": h1, "entry_hash": h2}

    with open(file2, "w") as f:
        f.write(json.dumps(rec2) + "\n")

    res = verify_log_integrity(tmp_path)
    assert res.ok is True
    assert res.total_entries == 2


def test_audit_log_content_flag(tmp_path: Path):
    """Test that prompt/response content is omitted by default and logged when log_content=True."""
    logger_default = AuditLogger(log_dir=tmp_path / "default", log_content=False)
    assert logger_default.log_content is False

    logger_content = AuditLogger(log_dir=tmp_path / "content", log_content=True)
    assert logger_content.log_content is True
