"""Integration tests for Phase 4 components (AnalyzerFactory, AnalysisPipeline)."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from src.core.factories.analyzer_factory import AnalyzerFactory
from src.core.pipeline.analysis_pipeline import AnalysisPipeline
from src.core.utils.config import Config
from src.core.analyzers.metadata import MetadataExtractor
from src.core.analyzers.content import ContentAnalyzer


class TestAnalyzerFactory:
    """Tests for AnalyzerFactory."""

    @pytest.fixture
    def default_config(self):
        """Create default config for testing."""
        config = Config()
        return config

    @pytest.fixture
    def yolo_config(self):
        """Create YOLO-enabled config."""
        config = Config(validate=False)
        config.set("analysis.ml_analysis", True)
        config.set("analysis.content_analysis", True)
        config.set("analysis.ml_models.use_yolo", True)
        config.set("analysis.ml_models.use_gpu", True)
        config.set("analysis.ml_models.batch_size", 16)
        config.set("processing.max_workers", 4)
        return config

    def test_create_metadata_extractor(self, default_config):
        """Factory creates MetadataExtractor."""
        factory = AnalyzerFactory(default_config)
        extractor = factory.create_metadata_extractor()

        assert isinstance(extractor, MetadataExtractor)

    def test_create_content_analyzer_enabled(self, default_config):
        """Factory creates ContentAnalyzer when enabled."""
        factory = AnalyzerFactory(default_config)
        analyzer = factory.create_content_analyzer()

        assert isinstance(analyzer, ContentAnalyzer)

    def test_create_content_analyzer_disabled(self):
        """Factory returns None when content analysis disabled."""
        config = Config(validate=False)
        config.set("analysis.content_analysis", False)
        factory = AnalyzerFactory(config)
        analyzer = factory.create_content_analyzer()

        assert analyzer is None

    def test_create_ml_analyzer_disabled(self):
        """Factory returns None when ML analysis disabled."""
        config = Config(validate=False)
        config.set("analysis.ml_analysis", False)
        factory = AnalyzerFactory(config)
        analyzer = factory.create_ml_analyzer()

        assert analyzer is None

    def test_create_all_analyzers(self, default_config):
        """Factory creates all analyzers."""
        factory = AnalyzerFactory(default_config)
        analyzers = factory.create_all_analyzers()

        assert "metadata" in analyzers
        assert "content" in analyzers
        assert "ml" in analyzers
        assert isinstance(analyzers["metadata"], MetadataExtractor)

    def test_yolo_priority(self):
        """YOLO analyzer takes priority when enabled."""
        # Use default config which has proper structure
        config = Config()
        config.set("analysis.ml_models.use_yolo", True)

        factory = AnalyzerFactory(config)

        # Mock the import to avoid loading actual models
        with patch.object(factory, "_create_yolo_analyzer") as mock:
            mock.return_value = Mock()
            analyzer = factory.create_ml_analyzer()

            mock.assert_called_once()


class TestAnalysisPipeline:
    """Tests for AnalysisPipeline."""

    @pytest.fixture
    def mock_analyzers(self):
        """Create mock analyzers for testing."""
        metadata = Mock(spec=MetadataExtractor)
        metadata.extract.return_value = {
            "file_name": "test.jpg",
            "file_path": "test1.jpg",
            "file_size": 1024,
        }

        content = Mock(spec=ContentAnalyzer)
        content.analyze.return_value = {
            "quality_score": 85.0,
            "is_blurry": False,
        }

        ml = Mock()
        ml.device = Mock()
        ml.device.type = "cpu"
        ml.analyze_batch.return_value = [
            {"primary_scene": "nature", "object_count": 2},
        ]

        return metadata, content, ml

    @pytest.fixture
    def sample_images(self):
        """Get sample images for testing."""
        test_paths = [
            Path("D:/Pictures/jpeg"),
            Path("C:/Users/kjfle/Pictures"),
        ]

        for test_path in test_paths:
            if test_path.exists():
                images = list(test_path.glob("*.jpg"))[:3]
                if images:
                    return [str(p) for p in images]

        pytest.skip("No test images available")

    def test_pipeline_init(self, mock_analyzers):
        """Pipeline initializes with analyzers."""
        metadata, content, ml = mock_analyzers
        config = Config()

        pipeline = AnalysisPipeline(
            metadata_extractor=metadata,
            content_analyzer=content,
            ml_analyzer=ml,
            config=config,
        )

        assert pipeline.metadata_extractor is metadata
        assert pipeline.content_analyzer is content
        assert pipeline.ml_analyzer is ml

    def test_pipeline_run_with_mocks(self, mock_analyzers):
        """Pipeline runs all phases with mocked analyzers."""
        metadata, content, ml = mock_analyzers
        config = Config(validate=False)
        config.set("processing.max_workers", 2)
        config.set("analysis.use_cache", False)

        pipeline = AnalysisPipeline(
            metadata_extractor=metadata,
            content_analyzer=content,
            ml_analyzer=ml,
            config=config,
        )

        results = pipeline.run(
            image_paths=["test1.jpg"],
            use_batch=True,
            show_progress=False,
        )

        assert len(results) == 1
        assert "file_name" in results[0]
        assert "quality_score" in results[0]
        assert "primary_scene" in results[0]

    def test_pipeline_without_content(self, mock_analyzers):
        """Pipeline works without content analyzer."""
        metadata, _, ml = mock_analyzers
        config = Config(validate=False)
        config.set("processing.max_workers", 2)
        config.set("analysis.use_cache", False)

        pipeline = AnalysisPipeline(
            metadata_extractor=metadata,
            content_analyzer=None,
            ml_analyzer=ml,
            config=config,
        )

        results = pipeline.run(
            image_paths=["test1.jpg"],
            use_batch=True,
            show_progress=False,
        )

        assert len(results) == 1
        assert "file_name" in results[0]

    def test_pipeline_without_ml(self, mock_analyzers):
        """Pipeline works without ML analyzer."""
        metadata, content, _ = mock_analyzers
        config = Config(validate=False)
        config.set("processing.max_workers", 2)
        config.set("analysis.use_cache", False)

        pipeline = AnalysisPipeline(
            metadata_extractor=metadata,
            content_analyzer=content,
            ml_analyzer=None,
            config=config,
        )

        results = pipeline.run(
            image_paths=["test1.jpg"],
            use_batch=True,
            show_progress=False,
        )

        assert len(results) == 1
        assert "file_name" in results[0]
        assert "quality_score" in results[0]


class TestIntegrationWithRealFiles:
    r"""Integration tests with real files (requires D:\Pictures\jpeg)."""

    @pytest.fixture
    def test_images(self):
        """Get real test images."""
        test_dir = Path("D:/Pictures/jpeg")
        if not test_dir.exists():
            pytest.skip("Test directory D:/Pictures/jpeg not found")

        images = list(test_dir.glob("*.jpg"))[:3]
        if not images:
            pytest.skip("No JPEG files found in test directory")

        return [str(p) for p in images]

    def test_metadata_extraction(self, test_images):
        """Test metadata extraction on real files."""
        extractor = MetadataExtractor()

        for path in test_images:
            result = extractor.extract(path)

            assert "file_path" in result or "image_path" in result
            assert "file_size" in result
            assert result["file_size"] > 0

    def test_content_analysis(self, test_images):
        """Test content analysis on real files."""
        analyzer = ContentAnalyzer(num_workers=2)

        for path in test_images:
            result = analyzer.analyze(path)

            assert "quality_score" in result
            assert 0 <= result["quality_score"] <= 100

    @pytest.mark.slow
    def test_full_pipeline(self, test_images):
        """Test full pipeline on real files (slow, requires GPU)."""
        config = Config()

        factory = AnalyzerFactory(config)
        analyzers = factory.create_all_analyzers()

        pipeline = AnalysisPipeline(
            metadata_extractor=analyzers["metadata"],
            content_analyzer=analyzers["content"],
            ml_analyzer=analyzers["ml"],
            config=config,
        )

        results = pipeline.run(
            image_paths=test_images,
            use_batch=True,
            show_progress=False,
        )

        assert len(results) == len(test_images)

        for result in results:
            assert "file_size" in result
