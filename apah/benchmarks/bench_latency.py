"""Apah Latency Benchmark: Time-To-First-Token (TTFT) and Inter-Token Latency (ITL) percentiles under load."""

import asyncio
import time
from typing import Dict, List, Tuple
from rich.console import Console
from rich.table import Table

from apah.benchmarks.bench_throughput import calculate_percentiles
from apah.cli.client import ApahClient

console = Console()


async def benchmark_latency_for_concurrency(
    client: ApahClient,
    model: str,
    concurrency: int,
    num_requests: int = 16,
) -> Dict[str, Tuple[float, float, float, float]]:
    """Measure TTFT and ITL distributions at p50, p90, p99, and avg percentiles."""
    prompt = "Enterprise refinery safety regulation document search request context details."
    ttfts: List[float] = []
    itls: List[float] = []

    sem = asyncio.Semaphore(concurrency)

    async def single_request():
        async with sem:
            start_t = time.perf_counter()
            first_token_t = None
            token_times = []

            async for chunk in client.chat_completion_stream_async(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=32,
                temperature=0.0,
            ):
                now = time.perf_counter()
                if first_token_t is None:
                    first_token_t = now
                    ttfts.append((first_token_t - start_t) * 1000.0)  # ms
                token_times.append(now)

            if len(token_times) > 1:
                diffs = [(token_times[i] - token_times[i - 1]) * 1000.0 for i in range(1, len(token_times))]
                itls.extend(diffs)

    tasks = [asyncio.create_task(single_request()) for _ in range(num_requests)]
    await asyncio.gather(*tasks, return_exceptions=True)

    ttft_stats = calculate_percentiles(ttfts)
    itl_stats = calculate_percentiles(itls)

    return {
        "concurrency": concurrency,
        "ttft": ttft_stats,  # (p50, p90, p99, avg) in ms
        "itl": itl_stats,    # (p50, p90, p99, avg) in ms
    }


def run_latency_benchmark(
    base_url: str = "http://localhost:11500",
    model: str = "default",
    concurrencies: Tuple[int, ...] = (1, 4, 8, 16, 32),
) -> List[Dict]:
    """Run latency benchmark suite reporting TTFT and ITL at p50/p90/p99 percentiles."""
    client = ApahClient(base_url=base_url)
    results = []

    table = Table(title=f"Apah Latency Percentiles (Model: {model}) [ms]")
    table.add_column("CONCURRENCY", style="cyan", justify="right")
    table.add_column("TTFT p50", style="green", justify="right")
    table.add_column("TTFT p90", style="yellow", justify="right")
    table.add_column("TTFT p99", style="red", justify="right")
    table.add_column("ITL p50", style="green", justify="right")
    table.add_column("ITL p90", style="yellow", justify="right")
    table.add_column("ITL p99", style="red", justify="right")

    for c in concurrencies:
        res = asyncio.run(benchmark_latency_for_concurrency(client, model, concurrency=c))
        results.append(res)
        ttft_p50, ttft_p90, ttft_p99, _ = res["ttft"]
        itl_p50, itl_p90, itl_p99, _ = res["itl"]

        table.add_row(
            str(c),
            f"{ttft_p50:.1f}",
            f"{ttft_p90:.1f}",
            f"{ttft_p99:.1f}",
            f"{itl_p50:.1f}",
            f"{itl_p90:.1f}",
            f"{itl_p99:.1f}",
        )

    console.print(table)
    return results


if __name__ == "__main__":
    run_latency_benchmark()
