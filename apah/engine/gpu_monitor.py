"""NVML wrapper for real-time NVIDIA GPU hardware metrics and caching."""

from dataclasses import dataclass
import logging
import time
from typing import List, Optional

logger = logging.getLogger("apah.engine.gpu_monitor")


@dataclass
class GPUStats:
    """Dataclass holding real-time telemetry metrics for a single NVIDIA GPU."""

    index: int
    name: str
    memory_used_mb: float
    memory_total_mb: float
    memory_free_mb: float
    utilization_pct: float
    temperature_c: float


class GPUMonitor:
    """Monitor providing cached real-time NVML hardware stats with non-fatal fallback."""

    _warned_unavailable: bool = False

    def __init__(self, cache_ttl_seconds: float = 0.5):
        self.cache_ttl_seconds = cache_ttl_seconds
        self._cached_stats: Optional[List[GPUStats]] = None
        self._last_cache_time: float = 0.0
        self.pynvml = None
        self._nvml_available = False

        self._init_nvml()

    def _init_nvml(self) -> None:
        """Initialize NVML bindings safely."""
        try:
            import pynvml
            pynvml.nvmlInit()
            self.pynvml = pynvml
            self._nvml_available = True
            logger.info("NVML GPU monitoring initialized successfully.")
        except Exception as e:
            if not GPUMonitor._warned_unavailable:
                logger.warning(
                    f"NVML GPU monitoring unavailable ({e}). Hardware stats (temperature, utilization) will be disabled."
                )
                GPUMonitor._warned_unavailable = True
            self.pynvml = None
            self._nvml_available = False

    def shutdown(self) -> None:
        """Shutdown NVML bindings cleanly."""
        if self._nvml_available and self.pynvml is not None:
            try:
                self.pynvml.nvmlShutdown()
                logger.info("NVML GPU monitoring shutdown cleanly.")
            except Exception as e:
                logger.warning(f"Error during NVML shutdown: {e}")
            self._nvml_available = False

    def get_device_count(self) -> int:
        """Return count of visible NVIDIA GPU devices."""
        if self._nvml_available and self.pynvml is not None:
            try:
                return int(self.pynvml.nvmlDeviceGetCount())
            except Exception:
                pass
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.device_count()
        except Exception:
            pass
        return 0

    def get_device_stats(self, device_index: int) -> Optional[GPUStats]:
        """Fetch hardware stats for a specific GPU device index."""
        if not self._nvml_available or self.pynvml is None:
            return None

        try:
            handle = self.pynvml.nvmlDeviceGetHandleByIndex(device_index)
            raw_name = self.pynvml.nvmlDeviceGetName(handle)
            name = raw_name.decode("utf-8") if isinstance(raw_name, bytes) else str(raw_name)

            mem_info = self.pynvml.nvmlDeviceGetMemoryInfo(handle)
            used_mb = float(mem_info.used) / (1024 * 1024)
            total_mb = float(mem_info.total) / (1024 * 1024)
            free_mb = float(mem_info.free) / (1024 * 1024)

            util_rates = self.pynvml.nvmlDeviceGetUtilizationRates(handle)
            utilization_pct = float(getattr(util_rates, "gpu", 0))

            temp_c = float(self.pynvml.nvmlDeviceGetTemperature(handle, self.pynvml.NVML_TEMPERATURE_GPU))

            return GPUStats(
                index=device_index,
                name=name,
                memory_used_mb=used_mb,
                memory_total_mb=total_mb,
                memory_free_mb=free_mb,
                utilization_pct=utilization_pct,
                temperature_c=temp_c,
            )
        except Exception as e:
            logger.warning(f"Failed to fetch NVML stats for GPU {device_index}: {e}")
            return None

    def get_all_device_stats(self) -> List[GPUStats]:
        """Return cached GPUStats list across all visible devices."""
        now = time.time()
        if self._cached_stats is not None and (now - self._last_cache_time) < self.cache_ttl_seconds:
            return self._cached_stats

        count = self.get_device_count()
        stats_list: List[GPUStats] = []
        for idx in range(count):
            st = self.get_device_stats(idx)
            if st is not None:
                stats_list.append(st)

        self._cached_stats = stats_list
        self._last_cache_time = now
        return stats_list
