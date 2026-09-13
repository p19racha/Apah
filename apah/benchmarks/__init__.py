"""Apah Benchmarking and Profiling Suite."""

from typing import Dict, List, NamedTuple


class MetricResult(NamedTuple):
    p50: float
    p90: float
    p99: float
    avg: float
