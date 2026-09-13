"""Apah Throughput Benchmark: prefill vs decode tokens/sec under varying concurrency levels."""

import asyncio
import time
from typing import Dict, List, Tuple
from rich.console import Console
from rich.table import Table

from apah.cli.client import ApahClient

console = Console()


def calculate_percentiles(values: List[float]) -> Tuple[float, float, float, float]:
    """Calculate p50, p90, p99, and avg for a list of numeric values."""
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    p50 = sorted_vals[int(n * 0.50)]
    p90 = sorted_vals[min(int(n * 0.90), n - 1)]
    p99 = sorted_vals[min(int(n * 0.99), n - 1)]
    avg = sum(sorted_vals) / n
    return p50, p90, p99, avg


async def benchmark_throughput_for_concurrency(
    client: ApahClient,
    model: str,
    concurrency: int,
    prompt_tokens_target: int = 128,
    gen_tokens_target: int = 64,
) -> Dict[str, float]:
    """Run concurrent completion requests and measure aggregate prefill & decode throughput."""
    prompt = "Refinery operational status metric check data " * (prompt_tokens_target // 6)

    async def single_request():
        start_t = time.perf_counter()
        first_token_t = None
        token_count = 0

        async for chunk in client.chat_completion_stream_async(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=gen_tokens_target,
            temperature=0.0,
        ):
            if first_token_t is None and chunk:
                first_token_t = time.perf_counter()
            if chunk:
                token_count += 1

        end_t = time.perf_counter()
        prefill_dur = (first_token_t - start_t) if first_token_t else (end_t - start_t)
        decode_dur = (end_t - first_token_t) if (first_token_t and end_t > first_token_t) else 0.001
        total_dur = end_t - start_t

        return {
            "prompt_tokens": prompt_tokens_target,
            "completion_tokens": token_count or gen_tokens_target,
            "prefill_dur": prefill_dur,
            "decode_dur": decode_dur,
            "total_dur": total_dur,
        }

    wall_start = time.perf_counter()
    tasks = [asyncio.create_task(single_request()) for _ in range(concurrency)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    wall_dur = time.perf_counter() - wall_start

    valid_results = [r for r in results if isinstance(r, dict)]
    total_prompt_toks = sum(r["prompt_tokens"] for r in valid_results)
    total_completion_toks = sum(r["completion_tokens"] for r in valid_results)

    avg_prefill_dur = sum(r["prefill_dur"] for r in valid_results) / len(valid_results) if valid_results else 1.0
    avg_decode_dur = sum(r["decode_dur"] for r in valid_results) / len(valid_results) if valid_results else 1.0

    prefill_throughput = total_prompt_toks / avg_prefill_dur if avg_prefill_dur > 0 else 0.0
    decode_throughput = total_completion_toks / avg_decode_dur if avg_decode_dur > 0 else 0.0
    aggregate_throughput = (total_prompt_toks + total_completion_toks) / wall_dur if wall_dur > 0 else 0.0

    return {
        "concurrency": concurrency,
        "prefill_tok_s": round(prefill_throughput, 2),
        "decode_tok_s": round(decode_throughput, 2),
        "aggregate_tok_s": round(aggregate_throughput, 2),
        "wall_time_s": round(wall_dur, 3),
    }


def run_throughput_benchmark(
    base_url: str = "http://localhost:11500",
    model: str = "default",
    concurrencies: Tuple[int, ...] = (1, 4, 8, 16, 32),
) -> List[Dict[str, float]]:
    """Run full throughput benchmark suite across specified concurrency levels."""
    client = ApahClient(base_url=base_url)
    results = []

    table = Table(title=f"Apah Throughput Benchmark (Model: {model})")
    table.add_column("CONCURRENCY", style="cyan", justify="right")
    table.add_column("PREFILL (TOK/S)", style="green", justify="right")
    table.add_column("DECODE (TOK/S)", style="yellow", justify="right")
    table.add_column("AGGREGATE (TOK/S)", style="bold magenta", justify="right")
    table.add_column("WALL TIME (S)", style="dim white", justify="right")

    for c in concurrencies:
        res = asyncio.run(benchmark_throughput_for_concurrency(client, model, concurrency=c))
        results.append(res)
        table.add_row(
            str(res["concurrency"]),
            f"{res['prefill_tok_s']:.2f}",
            f"{res['decode_tok_s']:.2f}",
            f"{res['aggregate_tok_s']:.2f}",
            f"{res['wall_time_s']:.3f}",
        )

    console.print(table)

    # Optional matplotlib plotting if installed
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        cs = [r["concurrency"] for r in results]
        ax.plot(cs, [r["prefill_tok_s"] for r in results], marker="o", label="Prefill tok/s")
        ax.plot(cs, [r["decode_tok_s"] for r in results], marker="s", label="Decode tok/s")
        ax.plot(cs, [r["aggregate_tok_s"] for r in results], marker="^", label="Aggregate tok/s", linewidth=2)
        ax.set_xlabel("Concurrency Level")
        ax.set_ylabel("Tokens / Second")
        ax.set_title("Apah Throughput Scaling")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig("apah_throughput_benchmark.png")
        console.print("[dim green]Saved throughput benchmark chart to apah_throughput_benchmark.png[/dim green]")
    except ImportError:
        pass

    return results


if __name__ == "__main__":
    run_throughput_benchmark()
