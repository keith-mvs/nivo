"""Performance metrics tracking for ML models and image processing.

Captures inference times, throughput, GPU utilization, and memory usage
to enable performance optimization and monitoring.
"""

import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from collections import defaultdict
from contextlib import contextmanager
import statistics

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class TimingRecord:
    """Single timing measurement."""
    name: str
    duration_ms: float
    batch_size: int = 1
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GPUMetrics:
    """GPU utilization snapshot."""
    memory_used_mb: float
    memory_total_mb: float
    memory_percent: float
    gpu_utilization: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ModelMetrics:
    """Aggregated metrics for a specific model/operation."""
    name: str
    total_calls: int = 0
    total_items: int = 0
    total_time_ms: float = 0.0
    min_time_ms: float = float('inf')
    max_time_ms: float = 0.0
    timing_samples: List[float] = field(default_factory=list)

    @property
    def avg_time_ms(self) -> float:
        """Average time per call in milliseconds."""
        if self.total_calls == 0:
            return 0.0
        return self.total_time_ms / self.total_calls

    @property
    def avg_time_per_item_ms(self) -> float:
        """Average time per item in milliseconds."""
        if self.total_items == 0:
            return 0.0
        return self.total_time_ms / self.total_items

    @property
    def throughput_per_sec(self) -> float:
        """Items processed per second."""
        if self.total_time_ms == 0:
            return 0.0
        return (self.total_items / self.total_time_ms) * 1000

    @property
    def std_dev_ms(self) -> float:
        """Standard deviation of timing samples."""
        if len(self.timing_samples) < 2:
            return 0.0
        return statistics.stdev(self.timing_samples)

    def record(self, duration_ms: float, batch_size: int = 1):
        """Record a timing measurement."""
        self.total_calls += 1
        self.total_items += batch_size
        self.total_time_ms += duration_ms
        self.min_time_ms = min(self.min_time_ms, duration_ms)
        self.max_time_ms = max(self.max_time_ms, duration_ms)
        self.timing_samples.append(duration_ms)

        # Keep only last 1000 samples for memory efficiency
        if len(self.timing_samples) > 1000:
            self.timing_samples = self.timing_samples[-1000:]


class PerformanceTracker:
    """Track performance metrics for ML models and processing operations.

    Usage:
        tracker = PerformanceTracker()

        # Time a single operation
        with tracker.time("yolo_inference", batch_size=16):
            results = model.predict(batch)

        # Get metrics
        metrics = tracker.get_metrics("yolo_inference")
        print(f"Avg: {metrics.avg_time_ms:.2f}ms, Throughput: {metrics.throughput_per_sec:.1f}/s")

        # Print summary
        tracker.print_summary()
    """

    def __init__(self, track_gpu: bool = True):
        """Initialize performance tracker.

        Args:
            track_gpu: Enable GPU metrics tracking (requires CUDA)
        """
        self._metrics: Dict[str, ModelMetrics] = {}
        self._timing_history: List[TimingRecord] = []
        self._gpu_history: List[GPUMetrics] = []
        self._track_gpu = track_gpu and TORCH_AVAILABLE and torch.cuda.is_available()
        self._start_time = datetime.now()

    @contextmanager
    def time(self, name: str, batch_size: int = 1, **metadata):
        """Context manager to time an operation.

        Args:
            name: Operation name (e.g., "yolo_inference", "clip_encoding")
            batch_size: Number of items in this batch
            **metadata: Additional metadata to store with timing

        Yields:
            None
        """
        start = time.perf_counter()

        # Record GPU state before
        gpu_before = self._get_gpu_metrics() if self._track_gpu else None

        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000

            # Record timing
            if name not in self._metrics:
                self._metrics[name] = ModelMetrics(name=name)

            self._metrics[name].record(elapsed_ms, batch_size)

            # Store detailed record
            record = TimingRecord(
                name=name,
                duration_ms=elapsed_ms,
                batch_size=batch_size,
                metadata=metadata,
            )
            self._timing_history.append(record)

            # Record GPU state after
            if self._track_gpu:
                gpu_after = self._get_gpu_metrics()
                if gpu_after:
                    self._gpu_history.append(gpu_after)

    def record(self, name: str, duration_ms: float, batch_size: int = 1):
        """Manually record a timing measurement.

        Args:
            name: Operation name
            duration_ms: Duration in milliseconds
            batch_size: Number of items processed
        """
        if name not in self._metrics:
            self._metrics[name] = ModelMetrics(name=name)

        self._metrics[name].record(duration_ms, batch_size)

    def get_metrics(self, name: str) -> Optional[ModelMetrics]:
        """Get metrics for a specific operation.

        Args:
            name: Operation name

        Returns:
            ModelMetrics or None if not found
        """
        return self._metrics.get(name)

    def get_all_metrics(self) -> Dict[str, ModelMetrics]:
        """Get all recorded metrics."""
        return self._metrics.copy()

    def _get_gpu_metrics(self) -> Optional[GPUMetrics]:
        """Get current GPU utilization metrics."""
        if not self._track_gpu:
            return None

        try:
            memory_used = torch.cuda.memory_allocated() / 1024 / 1024
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
            memory_percent = (memory_used / memory_total) * 100

            return GPUMetrics(
                memory_used_mb=memory_used,
                memory_total_mb=memory_total,
                memory_percent=memory_percent,
            )
        except Exception:
            return None

    def get_gpu_summary(self) -> Dict[str, float]:
        """Get GPU memory usage summary.

        Returns:
            Dictionary with avg, peak, min memory usage in MB
        """
        if not self._gpu_history:
            return {}

        memory_values = [g.memory_used_mb for g in self._gpu_history]

        return {
            "avg_memory_mb": statistics.mean(memory_values),
            "peak_memory_mb": max(memory_values),
            "min_memory_mb": min(memory_values),
            "samples": len(memory_values),
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary.

        Returns:
            Dictionary with all performance metrics
        """
        total_time = (datetime.now() - self._start_time).total_seconds()

        summary = {
            "session_duration_sec": total_time,
            "operations": {},
        }

        for name, metrics in self._metrics.items():
            summary["operations"][name] = {
                "total_calls": metrics.total_calls,
                "total_items": metrics.total_items,
                "avg_time_ms": round(metrics.avg_time_ms, 2),
                "avg_per_item_ms": round(metrics.avg_time_per_item_ms, 2),
                "throughput_per_sec": round(metrics.throughput_per_sec, 1),
                "min_time_ms": round(metrics.min_time_ms, 2),
                "max_time_ms": round(metrics.max_time_ms, 2),
                "std_dev_ms": round(metrics.std_dev_ms, 2),
            }

        if self._gpu_history:
            summary["gpu"] = self.get_gpu_summary()

        return summary

    def print_summary(self, title: str = "Performance Summary"):
        """Print formatted performance summary to console.

        Args:
            title: Summary title
        """
        print(f"\n{'='*60}")
        print(f" {title}")
        print(f"{'='*60}")

        summary = self.get_summary()
        print(f"Session duration: {summary['session_duration_sec']:.1f}s\n")

        # Operations table
        if summary["operations"]:
            print(f"{'Operation':<25} {'Calls':>8} {'Items':>8} {'Avg(ms)':>10} {'Thru(/s)':>10}")
            print("-" * 60)

            for name, stats in summary["operations"].items():
                print(
                    f"{name:<25} "
                    f"{stats['total_calls']:>8} "
                    f"{stats['total_items']:>8} "
                    f"{stats['avg_time_ms']:>10.2f} "
                    f"{stats['throughput_per_sec']:>10.1f}"
                )

        # GPU summary
        if "gpu" in summary:
            gpu = summary["gpu"]
            print(f"\nGPU Memory: avg={gpu['avg_memory_mb']:.0f}MB, "
                  f"peak={gpu['peak_memory_mb']:.0f}MB")

        print(f"{'='*60}\n")

    def reset(self):
        """Reset all metrics."""
        self._metrics.clear()
        self._timing_history.clear()
        self._gpu_history.clear()
        self._start_time = datetime.now()


# Global tracker instance
_global_tracker: Optional[PerformanceTracker] = None


def get_tracker() -> PerformanceTracker:
    """Get global performance tracker instance.

    Returns:
        Singleton PerformanceTracker instance
    """
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = PerformanceTracker()
    return _global_tracker


def reset_tracker():
    """Reset global performance tracker."""
    global _global_tracker
    if _global_tracker:
        _global_tracker.reset()


@contextmanager
def track_time(name: str, batch_size: int = 1, **metadata):
    """Convenience context manager using global tracker.

    Args:
        name: Operation name
        batch_size: Batch size
        **metadata: Additional metadata

    Yields:
        None
    """
    tracker = get_tracker()
    with tracker.time(name, batch_size, **metadata):
        yield
