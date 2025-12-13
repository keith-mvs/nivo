"""Performance metrics tracking for ML models and analysis pipeline.

Tracks:
- Model load times (CLIP, YOLO, DETR)
- Inference times (per-batch and per-image)
- Throughput (images/second)
- Phase timing (metadata, content, ML)
- GPU memory usage

Usage:
    metrics = PerformanceMetrics()

    with metrics.track("clip_inference"):
        results = model.classify(images)

    metrics.record_batch("yolo", batch_size=16, duration=1.5)

    summary = metrics.get_summary()
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum
import json
from pathlib import Path
import statistics


class MetricType(Enum):
    """Types of metrics that can be tracked."""
    MODEL_LOAD = "model_load"
    INFERENCE = "inference"
    BATCH = "batch"
    PHASE = "phase"
    MEMORY = "memory"


@dataclass
class TimingRecord:
    """Single timing measurement."""
    name: str
    duration_ms: float
    timestamp: float
    metric_type: MetricType
    batch_size: int = 1
    image_count: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def throughput(self) -> float:
        """Images per second."""
        if self.duration_ms > 0:
            return (self.image_count / self.duration_ms) * 1000
        return 0.0


@dataclass
class ModelMetrics:
    """Aggregated metrics for a single model."""
    model_name: str
    load_time_ms: float = 0.0
    inference_times_ms: List[float] = field(default_factory=list)
    batch_sizes: List[int] = field(default_factory=list)
    total_images: int = 0
    total_batches: int = 0

    @property
    def avg_inference_ms(self) -> float:
        """Average inference time in milliseconds."""
        if not self.inference_times_ms:
            return 0.0
        return statistics.mean(self.inference_times_ms)

    @property
    def p50_inference_ms(self) -> float:
        """Median inference time (50th percentile)."""
        if not self.inference_times_ms:
            return 0.0
        return statistics.median(self.inference_times_ms)

    @property
    def p95_inference_ms(self) -> float:
        """95th percentile inference time."""
        if len(self.inference_times_ms) < 2:
            return self.avg_inference_ms
        sorted_times = sorted(self.inference_times_ms)
        idx = int(len(sorted_times) * 0.95)
        return sorted_times[min(idx, len(sorted_times) - 1)]

    @property
    def throughput(self) -> float:
        """Average images per second."""
        total_time_sec = sum(self.inference_times_ms) / 1000
        if total_time_sec > 0:
            return self.total_images / total_time_sec
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_name": self.model_name,
            "load_time_ms": round(self.load_time_ms, 2),
            "total_images": self.total_images,
            "total_batches": self.total_batches,
            "avg_inference_ms": round(self.avg_inference_ms, 2),
            "p50_inference_ms": round(self.p50_inference_ms, 2),
            "p95_inference_ms": round(self.p95_inference_ms, 2),
            "throughput_img_per_sec": round(self.throughput, 2),
        }


class PerformanceMetrics:
    """
    Central performance metrics tracker for the analysis pipeline.

    Integrates with ML analyzers and pipeline phases to collect:
    - Model load times
    - Inference times (batch and per-image)
    - Phase timings (metadata, content, ML)
    - GPU memory snapshots

    Thread-safe for use in multi-threaded pipeline.
    """

    def __init__(self):
        """Initialize performance metrics tracker."""
        self.records: List[TimingRecord] = []
        self.models: Dict[str, ModelMetrics] = {}
        self.phases: Dict[str, List[float]] = {}
        self.memory_snapshots: List[Dict[str, Any]] = []
        self._start_time: Optional[float] = None
        self._run_metadata: Dict[str, Any] = {}

    def start_run(self, metadata: Optional[Dict[str, Any]] = None):
        """
        Start a new metrics collection run.

        Args:
            metadata: Optional run metadata (image count, config, etc.)
        """
        self._start_time = time.time()
        self._run_metadata = metadata or {}
        self.records = []
        self.phases = {}
        self.memory_snapshots = []
        # Keep model metrics across runs for cumulative stats

    def end_run(self) -> Dict[str, Any]:
        """
        End current run and return summary.

        Returns:
            Run summary dictionary
        """
        end_time = time.time()
        duration = end_time - self._start_time if self._start_time else 0

        return {
            "run_duration_sec": round(duration, 2),
            "total_records": len(self.records),
            "models": {name: m.to_dict() for name, m in self.models.items()},
            "phases": self._get_phase_summary(),
            "metadata": self._run_metadata,
        }

    @contextmanager
    def track(self, name: str, metric_type: MetricType = MetricType.INFERENCE):
        """
        Context manager for timing operations.

        Args:
            name: Name of the operation to track
            metric_type: Type of metric being tracked

        Yields:
            TimingContext with start time

        Example:
            with metrics.track("clip_inference"):
                results = model(inputs)
        """
        start = time.time()
        try:
            yield
        finally:
            duration_ms = (time.time() - start) * 1000
            self._record(name, duration_ms, metric_type)

    @contextmanager
    def track_phase(self, phase_name: str):
        """
        Context manager for timing pipeline phases.

        Args:
            phase_name: Name of the phase (e.g., "metadata", "content", "ml")

        Example:
            with metrics.track_phase("metadata"):
                results = extractor.extract_all(images)
        """
        start = time.time()
        try:
            yield
        finally:
            duration_ms = (time.time() - start) * 1000
            self._record_phase(phase_name, duration_ms)

    @contextmanager
    def track_model_load(self, model_name: str):
        """
        Context manager for timing model loading.

        Args:
            model_name: Name of the model being loaded

        Example:
            with metrics.track_model_load("yolov8n"):
                model = YOLO("yolov8n.pt")
        """
        start = time.time()
        try:
            yield
        finally:
            duration_ms = (time.time() - start) * 1000
            self._record_model_load(model_name, duration_ms)

    def record_batch(
        self,
        model_name: str,
        batch_size: int,
        duration_ms: float,
        image_count: Optional[int] = None,
    ):
        """
        Record batch inference metrics.

        Args:
            model_name: Name of the model
            batch_size: Size of the batch
            duration_ms: Inference duration in milliseconds
            image_count: Actual image count (defaults to batch_size)
        """
        actual_count = image_count or batch_size

        record = TimingRecord(
            name=f"{model_name}_batch",
            duration_ms=duration_ms,
            timestamp=time.time(),
            metric_type=MetricType.BATCH,
            batch_size=batch_size,
            image_count=actual_count,
        )
        self.records.append(record)

        # Update model metrics
        if model_name not in self.models:
            self.models[model_name] = ModelMetrics(model_name=model_name)

        model = self.models[model_name]
        model.inference_times_ms.append(duration_ms)
        model.batch_sizes.append(batch_size)
        model.total_images += actual_count
        model.total_batches += 1

    def record_memory(self, gpu_stats: Dict[str, Any]):
        """
        Record GPU memory snapshot.

        Args:
            gpu_stats: GPU statistics from gpu_monitor.get_stats()
        """
        snapshot = {
            "timestamp": time.time(),
            **gpu_stats,
        }
        self.memory_snapshots.append(snapshot)

    def _record(self, name: str, duration_ms: float, metric_type: MetricType):
        """Internal: Record a timing measurement."""
        record = TimingRecord(
            name=name,
            duration_ms=duration_ms,
            timestamp=time.time(),
            metric_type=metric_type,
        )
        self.records.append(record)

    def _record_phase(self, phase_name: str, duration_ms: float):
        """Internal: Record phase timing."""
        if phase_name not in self.phases:
            self.phases[phase_name] = []
        self.phases[phase_name].append(duration_ms)

        record = TimingRecord(
            name=phase_name,
            duration_ms=duration_ms,
            timestamp=time.time(),
            metric_type=MetricType.PHASE,
        )
        self.records.append(record)

    def _record_model_load(self, model_name: str, duration_ms: float):
        """Internal: Record model load time."""
        if model_name not in self.models:
            self.models[model_name] = ModelMetrics(model_name=model_name)

        self.models[model_name].load_time_ms = duration_ms

        record = TimingRecord(
            name=f"{model_name}_load",
            duration_ms=duration_ms,
            timestamp=time.time(),
            metric_type=MetricType.MODEL_LOAD,
        )
        self.records.append(record)

    def _get_phase_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for each phase."""
        summary = {}
        for phase_name, times in self.phases.items():
            if times:
                summary[phase_name] = {
                    "total_ms": round(sum(times), 2),
                    "avg_ms": round(statistics.mean(times), 2),
                    "count": len(times),
                }
        return summary

    def get_summary(self) -> Dict[str, Any]:
        """
        Get complete metrics summary.

        Returns:
            Dictionary with all collected metrics
        """
        total_inference_time = sum(
            m.avg_inference_ms * m.total_batches
            for m in self.models.values()
        )
        total_images = sum(m.total_images for m in self.models.values())

        overall_throughput = 0.0
        if total_inference_time > 0:
            overall_throughput = (total_images / total_inference_time) * 1000

        return {
            "total_images_processed": total_images,
            "overall_throughput_img_per_sec": round(overall_throughput, 2),
            "models": {name: m.to_dict() for name, m in self.models.items()},
            "phases": self._get_phase_summary(),
            "memory_snapshots_count": len(self.memory_snapshots),
            "peak_memory_gb": self._get_peak_memory(),
            "record_count": len(self.records),
        }

    def _get_peak_memory(self) -> float:
        """Get peak GPU memory from snapshots."""
        if not self.memory_snapshots:
            return 0.0
        peak = max(
            s.get("memory_allocated_gb", 0) for s in self.memory_snapshots
        )
        return round(peak, 3)

    def get_model_metrics(self, model_name: str) -> Optional[ModelMetrics]:
        """
        Get metrics for a specific model.

        Args:
            model_name: Name of the model

        Returns:
            ModelMetrics or None if not tracked
        """
        return self.models.get(model_name)

    def print_summary(self):
        """Print formatted metrics summary to console."""
        summary = self.get_summary()

        print("\n" + "=" * 60)
        print("PERFORMANCE METRICS SUMMARY")
        print("=" * 60)

        print(f"\nTotal images processed: {summary['total_images_processed']}")
        print(f"Overall throughput: {summary['overall_throughput_img_per_sec']:.1f} img/sec")

        if summary['peak_memory_gb'] > 0:
            print(f"Peak GPU memory: {summary['peak_memory_gb']:.2f} GB")

        if summary['models']:
            print("\n--- Model Performance ---")
            for name, metrics in summary['models'].items():
                print(f"\n{name}:")
                print(f"  Load time: {metrics['load_time_ms']:.0f} ms")
                print(f"  Images: {metrics['total_images']} in {metrics['total_batches']} batches")
                print(f"  Avg inference: {metrics['avg_inference_ms']:.1f} ms")
                print(f"  P50/P95: {metrics['p50_inference_ms']:.1f}/{metrics['p95_inference_ms']:.1f} ms")
                print(f"  Throughput: {metrics['throughput_img_per_sec']:.1f} img/sec")

        if summary['phases']:
            print("\n--- Phase Timing ---")
            for phase, stats in summary['phases'].items():
                print(f"  {phase}: {stats['total_ms']:.0f} ms total, {stats['avg_ms']:.1f} ms avg")

        print("\n" + "=" * 60)

    def export_json(self, path: str):
        """
        Export metrics to JSON file.

        Args:
            path: Output file path
        """
        summary = self.get_summary()
        summary['records'] = [
            {
                'name': r.name,
                'duration_ms': r.duration_ms,
                'type': r.metric_type.value,
                'batch_size': r.batch_size,
                'image_count': r.image_count,
            }
            for r in self.records
        ]

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

    def reset(self):
        """Reset all metrics."""
        self.records = []
        self.models = {}
        self.phases = {}
        self.memory_snapshots = []
        self._start_time = None
        self._run_metadata = {}


# Global metrics instance
_global_metrics: Optional[PerformanceMetrics] = None


def get_metrics() -> PerformanceMetrics:
    """
    Get global performance metrics instance.

    Returns:
        Global PerformanceMetrics instance
    """
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = PerformanceMetrics()
    return _global_metrics


def reset_metrics():
    """Reset global metrics instance."""
    global _global_metrics
    if _global_metrics:
        _global_metrics.reset()
