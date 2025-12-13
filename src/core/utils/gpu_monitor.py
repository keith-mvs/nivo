"""Real-time GPU monitoring utility."""

import torch
import threading
import time
from typing import Optional, Dict, Any
import subprocess
from ..utils.logging_config import get_logger

# Import interface
try:
    from ..interfaces.monitors import GPUMonitor as GPUMonitorInterface
except ImportError:
    # Fallback if interfaces not available
    GPUMonitorInterface = object



logger = get_logger(__name__)
class GPUMonitorImpl(GPUMonitorInterface):
    """Implementation of GPU monitoring with real-time stats."""

    def __init__(self, update_interval: float = 1.0):
        """
        Initialize GPU monitor.

        Args:
            update_interval: Update interval in seconds
        """
        self.update_interval = update_interval
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.stats: Dict[str, float] = {}
        self.has_gpu = torch.cuda.is_available()
        self.events: list = []
        self.peak_memory_mb: float = 0.0

    def start_monitoring(self, interval: float = 1.0):
        """Start monitoring in background thread (interface method)."""
        if interval != self.update_interval:
            self.update_interval = interval

        if not self.has_gpu:
            return

        if self.monitoring:
            return

        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop monitoring (interface method)."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)

    # Backwards compatibility aliases
    def start(self):
        """Start monitoring (legacy method)."""
        self.start_monitoring()

    def stop(self):
        """Stop monitoring (legacy method)."""
        self.stop_monitoring()

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            try:
                self._update_stats()
                time.sleep(self.update_interval)
            except Exception:
                pass

    def _update_stats(self):
        """Update GPU statistics."""
        if not self.has_gpu:
            return

        try:
            # PyTorch memory stats
            memory_allocated_mb = torch.cuda.memory_allocated() / 1e6
            self.stats['memory_allocated_gb'] = memory_allocated_mb / 1000
            self.stats['memory_reserved_gb'] = torch.cuda.memory_reserved() / 1e9
            self.stats['max_memory_allocated_gb'] = torch.cuda.max_memory_allocated() / 1e9

            # Track peak memory
            if memory_allocated_mb > self.peak_memory_mb:
                self.peak_memory_mb = memory_allocated_mb

            # Get GPU utilization from nvidia-smi
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory,temperature.gpu,power.draw',
                     '--format=csv,noheader,nounits'],
                    capture_output=True,
                    text=True,
                    timeout=1
                )
                if result.returncode == 0:
                    values = result.stdout.strip().split(',')
                    if len(values) >= 4:
                        self.stats['gpu_util'] = float(values[0].strip())
                        self.stats['mem_util'] = float(values[1].strip())
                        self.stats['temperature'] = float(values[2].strip())
                        self.stats['power_draw'] = float(values[3].strip())
            except:
                pass

        except Exception:
            pass

    def get_stats(self) -> Dict[str, Any]:
        """Get current GPU statistics (interface method)."""
        stats = self.stats.copy()
        stats['gpu_available'] = self.has_gpu
        return stats

    def record_event(self, event_name: str, metadata: Optional[Dict[str, Any]] = None):
        """Record a monitoring event (interface method)."""
        event = {
            'timestamp': time.time(),
            'event': event_name,
            'stats': self.get_stats(),
        }
        if metadata:
            event['metadata'] = metadata
        self.events.append(event)

    def get_peak_memory(self) -> float:
        """Get peak GPU memory usage in MB (interface method)."""
        return self.peak_memory_mb

    def reset_stats(self):
        """Reset monitoring statistics (interface method)."""
        self.stats = {}
        self.events = []
        self.peak_memory_mb = 0.0
        if self.has_gpu:
            torch.cuda.reset_peak_memory_stats()

    def get_status_string(self) -> str:
        """Get formatted status string."""
        if not self.has_gpu:
            return "CPU Mode"

        stats = self.get_stats()
        if not stats:
            return "GPU: Idle"

        parts = []

        # GPU utilization
        if 'gpu_util' in stats:
            parts.append(f"GPU: {stats['gpu_util']:.0f}%")

        # Memory
        if 'memory_allocated_gb' in stats:
            parts.append(f"Mem: {stats['memory_allocated_gb']:.1f}GB")

        # Temperature
        if 'temperature' in stats:
            parts.append(f"Temp: {stats['temperature']:.0f}°C")

        # Power
        if 'power_draw' in stats:
            parts.append(f"Power: {stats['power_draw']:.0f}W")

        return " | ".join(parts) if parts else "GPU: Active"

    def print_stats(self):
        """Print current GPU stats."""
        if not self.has_gpu:
            logger.warning("GPU: Not available")
            return

        stats = self.get_stats()
        if not stats:
            logger.info("GPU: No stats available")
            return

        logger.info("" + "=" * 60)
        logger.info("GPU Statistics")
        logger.info("=" * 60)

        if 'gpu_util' in stats:
            logger.info(f"GPU Utilization:    {stats['gpu_util']:.1f}%")
        if 'mem_util' in stats:
            logger.info(f"Memory Utilization: {stats['mem_util']:.1f}%")
        if 'memory_allocated_gb' in stats:
            logger.info(f"Memory Allocated:   {stats['memory_allocated_gb']:.2f} GB")
        if 'memory_reserved_gb' in stats:
            logger.info(f"Memory Reserved:    {stats['memory_reserved_gb']:.2f} GB")
        if 'temperature' in stats:
            logger.info(f"Temperature:        {stats['temperature']:.1f}°C")
        if 'power_draw' in stats:
            logger.info(f"Power Draw:         {stats['power_draw']:.1f} W")

        logger.info("=" * 60 + "\n")


# Backwards compatibility: alias the implementation class
GPUMonitor = GPUMonitorImpl


# Global monitor instance
_monitor: Optional[GPUMonitorImpl] = None


def get_monitor() -> GPUMonitorImpl:
    """
    Get global GPU monitor instance.

    Returns:
        Global GPU monitor instance
    """
    global _monitor
    if _monitor is None:
        _monitor = GPUMonitorImpl()
    return _monitor


def start_monitoring():
    """Start GPU monitoring."""
    get_monitor().start()


def stop_monitoring():
    """Stop GPU monitoring."""
    get_monitor().stop()


def get_status() -> str:
    """Get current GPU status string."""
    return get_monitor().get_status_string()
