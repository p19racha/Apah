"""Apah Memory Benchmark: Peak GPU memory scaling across mixed prompt lengths (Short, Medium, Long documents)."""

import math
from typing import Dict, List, Tuple
from rich.console import Console
from rich.table import Table

console = Console()


def estimate_padded_kv_memory_mb(
    num_sequences: int,
    max_seq_len: int,
    num_layers: int = 24,
    num_kv_heads: int = 14,
    head_dim: int = 64,
    bytes_per_elem: int = 2,
) -> float:
    """Calculate theoretical memory required by Phase 3 Padded KV Cache."""
    # Key + Value tensors per sequence padded to max_seq_len
    per_seq_bytes = 2 * num_layers * num_kv_heads * max_seq_len * head_dim * bytes_per_elem
    total_bytes = num_sequences * per_seq_bytes
    return total_bytes / (1024 * 1024)


def estimate_paged_kv_memory_mb(
    seq_lengths: List[int],
    block_size: int = 16,
    num_layers: int = 24,
    num_kv_heads: int = 14,
    head_dim: int = 64,
    bytes_per_elem: int = 2,
) -> float:
    """Calculate actual memory required by Phase 4 PagedAttention KV Cache blocks."""
    total_blocks = sum(math.ceil(length / block_size) for length in seq_lengths)
    block_bytes = 2 * block_size * num_layers * num_kv_heads * head_dim * bytes_per_elem
    return (total_blocks * block_bytes) / (1024 * 1024)


def run_memory_benchmark() -> List[Dict]:
    """Run memory footprint comparison benchmark across mixed prompt workloads."""
    workloads = [
        {"name": "Short Queries (128 tokens)", "concurrency": 16, "lengths": [128] * 16, "max_len": 128},
        {"name": "Medium Engineering Docs (512 tokens)", "concurrency": 16, "lengths": [512] * 16, "max_len": 512},
        {"name": "Long Refinery Manuals (2048 tokens)", "concurrency": 16, "lengths": [2048] * 16, "max_len": 2048},
        {"name": "Mixed Real-World (128, 512, 2048 mixed)", "concurrency": 16, "lengths": [128, 512, 2048, 256] * 4, "max_len": 2048},
    ]

    table = Table(title="Apah PagedAttention vs Padded KV Cache Memory Footprint Comparison")
    table.add_column("WORKLOAD PROFILE", style="cyan", no_wrap=True)
    table.add_column("CONCURRENCY", style="yellow", justify="right")
    table.add_column("PADDED KV MEM (MB)", style="red", justify="right")
    table.add_column("PAGED KV MEM (MB)", style="bold green", justify="right")
    table.add_column("MEMORY SAVINGS %", style="magenta", justify="right")

    results = []
    for w in workloads:
        padded_mb = estimate_padded_kv_memory_mb(w["concurrency"], w["max_len"])
        paged_mb = estimate_paged_kv_memory_mb(w["lengths"])
        savings_pct = ((padded_mb - paged_mb) / padded_mb * 100.0) if padded_mb > 0 else 0.0

        res = {
            "workload": w["name"],
            "concurrency": w["concurrency"],
            "padded_mb": round(padded_mb, 2),
            "paged_mb": round(paged_mb, 2),
            "savings_pct": round(savings_pct, 1),
        }
        results.append(res)
        table.add_row(
            w["name"],
            str(w["concurrency"]),
            f"{padded_mb:.2f} MB",
            f"{paged_mb:.2f} MB",
            f"[bold green]-{savings_pct:.1f}%[/bold green]",
        )

    console.print(table)
    return results


if __name__ == "__main__":
    run_memory_benchmark()
