"""Unit tests for GPUMonitor NVML wrapper, caching, and fallback behavior."""

from unittest.mock import MagicMock, patch
import pytest

from apah.engine.gpu_monitor import GPUMonitor, GPUStats


def test_gpu_monitor_parsing():
    """Test GPUMonitor parses NVML device hardware statistics correctly."""
    mock_nvml = MagicMock()
    mock_nvml.nvmlDeviceGetCount.return_value = 1

    mock_handle = MagicMock()
    mock_nvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
    mock_nvml.nvmlDeviceGetName.return_value = b"NVIDIA GeForce RTX 4070"

    mock_mem = MagicMock()
    mock_mem.used = 2 * 1024 * 1024 * 1024
    mock_mem.total = 12 * 1024 * 1024 * 1024
    mock_mem.free = 10 * 1024 * 1024 * 1024
    mock_nvml.nvmlDeviceGetMemoryInfo.return_value = mock_mem

    mock_rates = MagicMock()
    mock_rates.gpu = 45
    mock_nvml.nvmlDeviceGetUtilizationRates.return_value = mock_rates
    mock_nvml.nvmlDeviceGetTemperature.return_value = 55.0

    monitor = GPUMonitor(cache_ttl_seconds=0.5)
    monitor.pynvml = mock_nvml
    monitor._nvml_available = True

    stats = monitor.get_all_device_stats()
    assert len(stats) == 1
    gpu0 = stats[0]
    assert isinstance(gpu0, GPUStats)
    assert gpu0.index == 0
    assert gpu0.name == "NVIDIA GeForce RTX 4070"
    assert gpu0.memory_used_mb == 2048.0
    assert gpu0.memory_total_mb == 12288.0
    assert gpu0.utilization_pct == 45.0
    assert gpu0.temperature_c == 55.0


def test_gpu_monitor_fallback_when_nvml_fails():
    """Test GPUMonitor graceful non-fatal fallback when NVML initialization fails."""
    monitor = GPUMonitor(cache_ttl_seconds=0.5)
    monitor.pynvml = None
    monitor._nvml_available = False

    # Calling get_all_device_stats should return empty list or fallback without crashing
    stats = monitor.get_all_device_stats()
    assert isinstance(stats, list)


def test_gpu_monitor_caching():
    """Test 500ms caching window prevents redundant NVML calls."""
    mock_nvml = MagicMock()
    mock_nvml.nvmlDeviceGetCount.return_value = 1
    mock_handle = MagicMock()
    mock_nvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
    mock_nvml.nvmlDeviceGetName.return_value = "Mock GPU"

    mock_mem = MagicMock(used=1024*1024*1024, total=8*1024*1024*1024, free=7*1024*1024*1024)
    mock_nvml.nvmlDeviceGetMemoryInfo.return_value = mock_mem
    mock_nvml.nvmlDeviceGetUtilizationRates.return_value = MagicMock(gpu=20)
    mock_nvml.nvmlDeviceGetTemperature.return_value = 40.0

    monitor = GPUMonitor(cache_ttl_seconds=0.5)
    monitor.pynvml = mock_nvml
    monitor._nvml_available = True

    # Call get_all_device_stats 5 times in rapid succession
    for _ in range(5):
        monitor.get_all_device_stats()

    # Under 500ms TTL, nvmlDeviceGetHandleByIndex should only be invoked once
    assert mock_nvml.nvmlDeviceGetHandleByIndex.call_count == 1
