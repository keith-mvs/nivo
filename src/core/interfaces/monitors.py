"""GPU monitoring interfaces and implementations."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class GPUMonitor(ABC):
    """Abstract interface for GPU monitoring."""

    @abstractmethod
    def start_monitoring(self, interval: float = 1.0):
        """
        Start background GPU monitoring.

        Args:
            interval: Monitoring interval in seconds
        """
        pass

    @abstractmethod
    def stop_monitoring(self):
        """Stop background monitoring."""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Get current GPU statistics.

        Returns:
            Dictionary with GPU stats (memory, utilization, temperature, etc.)
        """
        pass

    @abstractmethod
    def record_event(self, event_name: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Record a monitoring event.

        Args:
            event_name: Name of the event
            metadata: Optional event metadata
        """
        pass

    @abstractmethod
    def get_peak_memory(self) -> float:
        """
        Get peak GPU memory usage in MB.

        Returns:
            Peak memory usage
        """
        pass

    @abstractmethod
    def reset_stats(self):
        """Reset monitoring statistics."""
        pass


class NullGPUMonitor(GPUMonitor):
    """Null object implementation for GPU monitor (CPU mode or testing)."""

    def start_monitoring(self, interval: float = 1.0):
        """No-op: CPU mode doesn't monitor GPU."""
        pass

    def stop_monitoring(self):
        """No-op: CPU mode doesn't monitor GPU."""
        pass

    def get_stats(self) -> Dict[str, Any]:
        """Return empty stats for CPU mode."""
        return {
            'gpu_available': False,
            'memory_used_mb': 0.0,
            'memory_total_mb': 0.0,
            'utilization_percent': 0.0,
            'temperature_c': 0.0,
        }

    def record_event(self, event_name: str, metadata: Optional[Dict[str, Any]] = None):
        """No-op: CPU mode doesn't record events."""
        pass

    def get_peak_memory(self) -> float:
        """Return 0 for CPU mode."""
        return 0.0

    def reset_stats(self):
        """No-op: CPU mode has no stats."""
        pass
