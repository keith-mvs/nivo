"""Unit tests for performance metrics module."""

import pytest
import time
from src.core.utils.performance_metrics import (
    PerformanceMetrics,
    ModelMetrics,
    TimingRecord,
    MetricType,
    get_metrics,
    reset_metrics,
)


class TestTimingRecord:
    """Tests for TimingRecord dataclass."""

    def test_create_record(self):
        """Record stores timing data correctly."""
        record = TimingRecord(
            name="test_op",
            duration_ms=100.5,
            timestamp=time.time(),
            metric_type=MetricType.INFERENCE,
        )

        assert record.name == "test_op"
        assert record.duration_ms == 100.5
        assert record.metric_type == MetricType.INFERENCE
        assert record.batch_size == 1
        assert record.image_count == 1

    def test_throughput_calculation(self):
        """Throughput is calculated correctly."""
        record = TimingRecord(
            name="batch",
            duration_ms=1000,  # 1 second
            timestamp=time.time(),
            metric_type=MetricType.BATCH,
            image_count=10,
        )

        assert record.throughput == 10.0  # 10 images per second

    def test_throughput_zero_duration(self):
        """Throughput returns 0 for zero duration."""
        record = TimingRecord(
            name="instant",
            duration_ms=0,
            timestamp=time.time(),
            metric_type=MetricType.INFERENCE,
        )

        assert record.throughput == 0.0


class TestModelMetrics:
    """Tests for ModelMetrics dataclass."""

    def test_create_model_metrics(self):
        """ModelMetrics initializes correctly."""
        metrics = ModelMetrics(model_name="yolov8n")

        assert metrics.model_name == "yolov8n"
        assert metrics.load_time_ms == 0.0
        assert metrics.total_images == 0
        assert metrics.total_batches == 0

    def test_avg_inference_time(self):
        """Average inference time is calculated correctly."""
        metrics = ModelMetrics(model_name="clip")
        metrics.inference_times_ms = [100, 150, 200]

        assert metrics.avg_inference_ms == 150.0

    def test_avg_inference_empty(self):
        """Average returns 0 for no data."""
        metrics = ModelMetrics(model_name="clip")

        assert metrics.avg_inference_ms == 0.0

    def test_p50_inference_time(self):
        """P50 (median) is calculated correctly."""
        metrics = ModelMetrics(model_name="clip")
        metrics.inference_times_ms = [100, 150, 200, 250, 300]

        assert metrics.p50_inference_ms == 200.0

    def test_p95_inference_time(self):
        """P95 is calculated correctly."""
        metrics = ModelMetrics(model_name="clip")
        metrics.inference_times_ms = list(range(1, 101))  # 1-100

        # P95 should be around 95
        assert metrics.p95_inference_ms >= 90

    def test_throughput(self):
        """Throughput is calculated correctly."""
        metrics = ModelMetrics(model_name="yolo")
        metrics.inference_times_ms = [100, 100, 100]  # 300ms total
        metrics.total_images = 30

        # 30 images / 0.3 seconds = 100 img/sec
        assert metrics.throughput == 100.0

    def test_to_dict(self):
        """to_dict serializes all fields."""
        metrics = ModelMetrics(model_name="test")
        metrics.load_time_ms = 500.0
        metrics.total_images = 100
        metrics.total_batches = 10

        result = metrics.to_dict()

        assert result["model_name"] == "test"
        assert result["load_time_ms"] == 500.0
        assert result["total_images"] == 100
        assert result["total_batches"] == 10
        assert "avg_inference_ms" in result
        assert "throughput_img_per_sec" in result


class TestPerformanceMetrics:
    """Tests for PerformanceMetrics class."""

    @pytest.fixture
    def metrics(self):
        """Create fresh metrics instance."""
        return PerformanceMetrics()

    def test_init(self, metrics):
        """Metrics initializes with empty state."""
        assert len(metrics.records) == 0
        assert len(metrics.models) == 0
        assert len(metrics.phases) == 0

    def test_track_context_manager(self, metrics):
        """track() context manager records timing."""
        with metrics.track("test_operation"):
            time.sleep(0.01)  # 10ms

        assert len(metrics.records) == 1
        assert metrics.records[0].name == "test_operation"
        assert metrics.records[0].duration_ms >= 10

    def test_track_phase(self, metrics):
        """track_phase() records phase timing."""
        with metrics.track_phase("metadata"):
            time.sleep(0.01)

        assert "metadata" in metrics.phases
        assert len(metrics.phases["metadata"]) == 1
        assert metrics.phases["metadata"][0] >= 10

    def test_track_model_load(self, metrics):
        """track_model_load() records load time."""
        with metrics.track_model_load("yolov8n"):
            time.sleep(0.01)

        assert "yolov8n" in metrics.models
        assert metrics.models["yolov8n"].load_time_ms >= 10

    def test_record_batch(self, metrics):
        """record_batch() tracks batch inference."""
        metrics.record_batch("clip", batch_size=16, duration_ms=150)

        assert "clip" in metrics.models
        model = metrics.models["clip"]
        assert model.total_images == 16
        assert model.total_batches == 1
        assert 150 in model.inference_times_ms

    def test_record_batch_custom_image_count(self, metrics):
        """record_batch() with custom image count."""
        metrics.record_batch("yolo", batch_size=16, duration_ms=100, image_count=12)

        model = metrics.models["yolo"]
        assert model.total_images == 12  # Uses image_count, not batch_size

    def test_record_memory(self, metrics):
        """record_memory() stores GPU snapshots."""
        gpu_stats = {"memory_allocated_gb": 1.5, "gpu_util": 80}
        metrics.record_memory(gpu_stats)

        assert len(metrics.memory_snapshots) == 1
        assert metrics.memory_snapshots[0]["memory_allocated_gb"] == 1.5

    def test_get_summary(self, metrics):
        """get_summary() returns complete metrics."""
        metrics.record_batch("yolo", batch_size=8, duration_ms=100)
        metrics.record_batch("clip", batch_size=8, duration_ms=50)

        with metrics.track_phase("ml"):
            pass

        summary = metrics.get_summary()

        assert "total_images_processed" in summary
        assert summary["total_images_processed"] == 16
        assert "models" in summary
        assert "yolo" in summary["models"]
        assert "clip" in summary["models"]
        assert "phases" in summary

    def test_start_end_run(self, metrics):
        """start_run/end_run track run duration."""
        metrics.start_run({"image_count": 100})
        time.sleep(0.01)
        summary = metrics.end_run()

        assert summary["run_duration_sec"] >= 0.01
        assert summary["metadata"]["image_count"] == 100

    def test_reset(self, metrics):
        """reset() clears all data."""
        metrics.record_batch("yolo", batch_size=8, duration_ms=100)
        metrics.record_memory({"memory_allocated_gb": 1.0})

        metrics.reset()

        assert len(metrics.records) == 0
        assert len(metrics.models) == 0
        assert len(metrics.memory_snapshots) == 0

    def test_get_model_metrics(self, metrics):
        """get_model_metrics() retrieves specific model."""
        metrics.record_batch("yolo", batch_size=8, duration_ms=100)

        model = metrics.get_model_metrics("yolo")
        assert model is not None
        assert model.model_name == "yolo"

        missing = metrics.get_model_metrics("nonexistent")
        assert missing is None

    def test_multiple_batches_same_model(self, metrics):
        """Multiple batches aggregate correctly."""
        metrics.record_batch("yolo", batch_size=16, duration_ms=100)
        metrics.record_batch("yolo", batch_size=16, duration_ms=120)
        metrics.record_batch("yolo", batch_size=8, duration_ms=60)

        model = metrics.models["yolo"]
        assert model.total_batches == 3
        assert model.total_images == 40  # 16 + 16 + 8
        assert len(model.inference_times_ms) == 3

    def test_phase_summary(self, metrics):
        """Phase summary aggregates timing."""
        with metrics.track_phase("metadata"):
            time.sleep(0.01)
        with metrics.track_phase("metadata"):
            time.sleep(0.01)

        summary = metrics.get_summary()
        phase_stats = summary["phases"]["metadata"]

        assert phase_stats["count"] == 2
        assert phase_stats["total_ms"] >= 20

    def test_peak_memory(self, metrics):
        """Peak memory is tracked correctly."""
        metrics.record_memory({"memory_allocated_gb": 1.0})
        metrics.record_memory({"memory_allocated_gb": 2.5})
        metrics.record_memory({"memory_allocated_gb": 1.5})

        summary = metrics.get_summary()
        assert summary["peak_memory_gb"] == 2.5

    def test_export_json(self, metrics, tmp_path):
        """export_json() creates valid JSON file."""
        metrics.record_batch("yolo", batch_size=8, duration_ms=100)

        output_path = tmp_path / "metrics.json"
        metrics.export_json(str(output_path))

        assert output_path.exists()

        import json
        with open(output_path) as f:
            data = json.load(f)

        assert "models" in data
        assert "records" in data


class TestGlobalMetrics:
    """Tests for global metrics functions."""

    def test_get_metrics_singleton(self):
        """get_metrics() returns same instance."""
        reset_metrics()
        m1 = get_metrics()
        m2 = get_metrics()

        assert m1 is m2

    def test_reset_metrics(self):
        """reset_metrics() clears global instance."""
        metrics = get_metrics()
        metrics.record_batch("test", batch_size=4, duration_ms=50)

        reset_metrics()

        assert len(get_metrics().records) == 0


class TestMetricType:
    """Tests for MetricType enum."""

    def test_metric_types_exist(self):
        """All expected metric types exist."""
        assert MetricType.MODEL_LOAD.value == "model_load"
        assert MetricType.INFERENCE.value == "inference"
        assert MetricType.BATCH.value == "batch"
        assert MetricType.PHASE.value == "phase"
        assert MetricType.MEMORY.value == "memory"
