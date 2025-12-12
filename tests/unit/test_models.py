"""Unit tests for domain models."""

import pytest
from datetime import datetime
from src.core.models import (
    ImageMetadata,
    ContentAnalysis,
    MLAnalysis,
    ImageAnalysisResult,
    RenameResult,
    TagEmbedResult,
    DuplicationResult,
    FormatConversionResult,
)


class TestImageMetadata:
    """Test ImageMetadata domain model."""

    def test_minimal_creation(self):
        """Test creating ImageMetadata with minimal required fields."""
        metadata = ImageMetadata(
            file_name="test.jpg",
            file_path="C:\\Users\\test\\test.jpg",
            file_size=1024000,
            width=1920,
            height=1080,
        )

        assert metadata.file_name == "test.jpg"
        assert metadata.width == 1920
        assert metadata.height == 1080
        assert metadata.megapixels is None

    def test_full_creation(self):
        """Test creating ImageMetadata with all fields."""
        dt = datetime(2024, 1, 15, 14, 30, 0)

        metadata = ImageMetadata(
            file_name="IMG_1234.jpg",
            file_path="C:\\Pictures\\IMG_1234.jpg",
            file_size=5242880,
            width=4032,
            height=3024,
            format="JPEG",
            mode="RGB",
            megapixels=12.19,
            datetime_original=dt,
            datetime_modified=dt,
            latitude=37.7749,
            longitude=-122.4194,
            altitude=50.0,
            make="Canon",
            model="EOS R5",
            f_number=2.8,
            iso_speed_ratings=800,
        )

        assert metadata.megapixels == 12.19
        assert metadata.datetime_original == dt
        assert metadata.latitude == 37.7749
        assert metadata.make == "Canon"

    def test_to_dict(self):
        """Test converting to dictionary with datetime serialization."""
        dt = datetime(2024, 1, 15, 14, 30, 0)

        metadata = ImageMetadata(
            file_name="test.jpg",
            file_path="C:\\test.jpg",
            file_size=1024,
            width=1920,
            height=1080,
            datetime_original=dt,
        )

        data = metadata.to_dict()

        assert data['file_name'] == "test.jpg"
        assert data['datetime_original'] == dt.isoformat()
        assert isinstance(data['datetime_original'], str)

    def test_from_dict(self):
        """Test creating from dictionary with datetime parsing."""
        data = {
            'file_name': "test.jpg",
            'file_path': "C:\\test.jpg",
            'file_size': 1024,
            'width': 1920,
            'height': 1080,
            'datetime_original': "2024-01-15T14:30:00",
            'extra_field': "should_be_ignored",
        }

        metadata = ImageMetadata.from_dict(data)

        assert metadata.file_name == "test.jpg"
        assert isinstance(metadata.datetime_original, datetime)
        assert metadata.datetime_original.year == 2024

    def test_roundtrip(self):
        """Test to_dict -> from_dict roundtrip."""
        dt = datetime(2024, 1, 15, 14, 30, 0)

        original = ImageMetadata(
            file_name="test.jpg",
            file_path="C:\\test.jpg",
            file_size=1024,
            width=1920,
            height=1080,
            datetime_original=dt,
            latitude=37.7749,
        )

        data = original.to_dict()
        restored = ImageMetadata.from_dict(data)

        assert restored.file_name == original.file_name
        assert restored.latitude == original.latitude
        assert restored.datetime_original == original.datetime_original


class TestContentAnalysis:
    """Test ContentAnalysis domain model."""

    def test_minimal_creation(self):
        """Test creating ContentAnalysis with defaults."""
        analysis = ContentAnalysis()

        assert analysis.phash is None
        assert analysis.quality_score is None
        assert analysis.error is None

    def test_full_creation(self):
        """Test creating ContentAnalysis with all fields."""
        analysis = ContentAnalysis(
            phash="abc123def456",
            average_hash="xyz789",
            sharpness_score=450.5,
            is_blurry=False,
            sharpness_level="sharp",
            quality_score=85.3,
            noise_level=12.4,
            dynamic_range=255,
            brightness=128.5,
            contrast=65.2,
            exposure="well_exposed",
            dominant_colors=[
                {"rgb": [120, 80, 40], "percentage": 45.2},
                {"rgb": [200, 190, 180], "percentage": 30.1},
            ],
            average_color=[150, 140, 130],
            color_temperature="warm",
            warmth_score=0.15,
        )

        assert analysis.sharpness_score == 450.5
        assert analysis.is_blurry is False
        assert len(analysis.dominant_colors) == 2
        assert analysis.color_temperature == "warm"

    def test_to_dict(self):
        """Test converting to dictionary."""
        analysis = ContentAnalysis(
            phash="abc123",
            quality_score=85.0,
        )

        data = analysis.to_dict()

        assert data['phash'] == "abc123"
        assert data['quality_score'] == 85.0

    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            'phash': "abc123",
            'quality_score': 85.0,
            'dominant_colors': [{"rgb": [120, 80, 40], "percentage": 45.2}],
        }

        analysis = ContentAnalysis.from_dict(data)

        assert analysis.phash == "abc123"
        assert len(analysis.dominant_colors) == 1


class TestMLAnalysis:
    """Test MLAnalysis domain model."""

    def test_minimal_creation(self):
        """Test creating MLAnalysis with defaults."""
        analysis = MLAnalysis()

        assert analysis.primary_scene is None
        assert analysis.scene_scores == {}
        assert analysis.objects == []
        assert analysis.tags == []

    def test_full_creation(self):
        """Test creating MLAnalysis with all fields."""
        analysis = MLAnalysis(
            primary_scene="outdoor",
            scene_scores={"outdoor": 0.95, "nature": 0.87, "landscape": 0.75},
            objects=[
                {"label": "person", "confidence": 0.92, "bbox": [10, 20, 100, 200]},
                {"label": "dog", "confidence": 0.88, "bbox": [150, 100, 250, 300]},
            ],
            object_counts={"person": 1, "dog": 1},
            tags=["outdoor", "nature", "person", "dog"],
            scene_model="openai/clip-vit-base-patch32",
            object_model="yolov8n.pt",
            inference_time=1.25,
            gpu_memory_used=1024.5,
        )

        assert analysis.primary_scene == "outdoor"
        assert len(analysis.objects) == 2
        assert analysis.object_counts["person"] == 1
        assert len(analysis.tags) == 4
        assert analysis.inference_time == 1.25

    def test_from_dict_with_defaults(self):
        """Test from_dict ensures default factories."""
        data = {
            'primary_scene': "outdoor",
            # Omit lists/dicts to test defaults
        }

        analysis = MLAnalysis.from_dict(data)

        assert analysis.primary_scene == "outdoor"
        assert analysis.scene_scores == {}
        assert analysis.objects == []
        assert analysis.object_counts == {}
        assert analysis.tags == []


class TestImageAnalysisResult:
    """Test ImageAnalysisResult combined model."""

    def test_minimal_creation(self):
        """Test creating ImageAnalysisResult with minimal data."""
        result = ImageAnalysisResult(
            image_path="C:\\test.jpg"
        )

        assert result.image_path == "C:\\test.jpg"
        assert result.metadata is None
        assert result.content is None
        assert result.ml_vision is None
        assert result.analysis_complete is False
        assert result.phases_completed == []

    def test_full_creation(self):
        """Test creating ImageAnalysisResult with all phases."""
        metadata = ImageMetadata(
            file_name="test.jpg",
            file_path="C:\\test.jpg",
            file_size=1024,
            width=1920,
            height=1080,
        )

        content = ContentAnalysis(
            phash="abc123",
            quality_score=85.0,
        )

        ml_vision = MLAnalysis(
            primary_scene="outdoor",
            tags=["outdoor", "nature"],
        )

        result = ImageAnalysisResult(
            image_path="C:\\test.jpg",
            metadata=metadata,
            content=content,
            ml_vision=ml_vision,
        )

        assert result.metadata.file_name == "test.jpg"
        assert result.content.phash == "abc123"
        assert result.ml_vision.primary_scene == "outdoor"

    def test_add_error(self):
        """Test adding errors."""
        result = ImageAnalysisResult(image_path="C:\\test.jpg")

        result.add_error("Test error 1")
        result.add_error("Test error 2")

        assert len(result.errors) == 2
        assert "Test error 1" in result.errors

    def test_mark_phase_complete(self):
        """Test marking phases as complete."""
        result = ImageAnalysisResult(image_path="C:\\test.jpg")

        result.mark_phase_complete("metadata")
        assert "metadata" in result.phases_completed
        assert result.analysis_complete is False

        result.mark_phase_complete("content")
        result.mark_phase_complete("ml_vision")

        assert result.analysis_complete is True

    def test_to_dict(self):
        """Test converting to dictionary with nested conversions."""
        metadata = ImageMetadata(
            file_name="test.jpg",
            file_path="C:\\test.jpg",
            file_size=1024,
            width=1920,
            height=1080,
        )

        result = ImageAnalysisResult(
            image_path="C:\\test.jpg",
            metadata=metadata,
        )

        data = result.to_dict()

        assert data['image_path'] == "C:\\test.jpg"
        assert data['metadata']['file_name'] == "test.jpg"
        assert data['content'] is None

    def test_from_dict(self):
        """Test creating from dictionary with nested parsing."""
        data = {
            'image_path': "C:\\test.jpg",
            'metadata': {
                'file_name': "test.jpg",
                'file_path': "C:\\test.jpg",
                'file_size': 1024,
                'width': 1920,
                'height': 1080,
            },
            'content': {
                'phash': "abc123",
            },
            'phases_completed': ["metadata", "content"],
        }

        result = ImageAnalysisResult.from_dict(data)

        assert result.image_path == "C:\\test.jpg"
        assert isinstance(result.metadata, ImageMetadata)
        assert result.metadata.file_name == "test.jpg"
        assert isinstance(result.content, ContentAnalysis)
        assert result.content.phash == "abc123"
        assert len(result.phases_completed) == 2


class TestProcessorResults:
    """Test processor result models."""

    def test_rename_result(self):
        """Test RenameResult model."""
        result = RenameResult(
            original_path="C:\\old.jpg",
            original_name="old.jpg",
            new_path="C:\\new.jpg",
            new_name="new.jpg",
            success=True,
            renamed=True,
            renamed_at=datetime.now(),
        )

        assert result.original_name == "old.jpg"
        assert result.new_name == "new.jpg"
        assert result.success is True

        data = result.to_dict()
        assert 'renamed_at' in data
        assert isinstance(data['renamed_at'], str)

    def test_tag_embed_result(self):
        """Test TagEmbedResult model."""
        result = TagEmbedResult(
            file_path="C:\\test.jpg",
            tags_embedded=["outdoor", "nature", "landscape"],
            tag_count=3,
            iptc_keywords=True,
            iptc_caption=True,
            caption_text="Outdoor nature scene with beautiful landscape",
            success=True,
            modified=True,
        )

        assert len(result.tags_embedded) == 3
        assert result.tag_count == 3
        assert result.iptc_keywords is True
        assert result.success is True

    def test_duplication_result(self):
        """Test DuplicationResult model."""
        result = DuplicationResult(
            file_path="C:\\test.jpg",
            file_hash="abc123",
            is_duplicate=True,
            duplicate_of="C:\\original.jpg",
            duplicates_found=["C:\\copy1.jpg", "C:\\copy2.jpg"],
            duplicate_count=2,
        )

        assert result.is_duplicate is True
        assert len(result.duplicates_found) == 2
        assert result.duplicate_count == 2

    def test_format_conversion_result(self):
        """Test FormatConversionResult model."""
        result = FormatConversionResult(
            original_path="C:\\test.png",
            original_format="PNG",
            original_size=5242880,
            converted_path="C:\\test.jpg",
            converted_format="JPEG",
            converted_size=1048576,
            success=True,
            converted=True,
            conversion_time=0.5,
            size_reduction=80.0,
        )

        assert result.success is True
        assert result.converted is True
        assert result.size_reduction == 80.0
        assert result.conversion_time == 0.5
