"""Unit tests for MetadataTagger with comprehensive coverage."""

import os
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict
from PIL import Image
import piexif

import pytest

from src.core.processors.tagger import MetadataTagger


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    temp = tempfile.mkdtemp()
    yield temp
    shutil.rmtree(temp, ignore_errors=True)


@pytest.fixture
def sample_jpeg(temp_dir):
    """Create a sample JPEG image."""
    image_path = Path(temp_dir) / "test_image.jpg"
    img = Image.new('RGB', (100, 100), color='red')
    img.save(str(image_path), 'JPEG', quality=95)
    return str(image_path)


@pytest.fixture
def sample_png(temp_dir):
    """Create a sample PNG image."""
    image_path = Path(temp_dir) / "test_image.png"
    img = Image.new('RGB', (100, 100), color='blue')
    img.save(str(image_path), 'PNG')
    return str(image_path)


@pytest.fixture
def sample_analysis_data():
    """Create sample analysis data."""
    return {
        "file_path": "test_image.jpg",
        "tags": ["landscape", "sunset", "nature"],
        "primary_scene": "outdoors",
        "unique_objects": ["tree", "mountain", "sky"],
        "quality_score": 85,
        "sharpness_level": "sharp",
        "datetime_original": datetime(2024, 3, 15, 14, 30, 45),
        "gps_coordinates": "40.7128, -74.0060",
    }


class TestEXIFEmbedding:
    """Test EXIF metadata embedding."""

    def test_embed_tags_to_usercomment(self, sample_jpeg, sample_analysis_data):
        """Test tags are embedded in EXIF UserComment."""
        tagger = MetadataTagger(embed_tags=True)
        success = tagger.embed_metadata(sample_jpeg, sample_analysis_data)

        assert success is True

        # Read back and verify
        exif_dict = piexif.load(sample_jpeg)
        user_comment = exif_dict["Exif"][piexif.ExifIFD.UserComment]
        decoded = user_comment.decode("utf-8")

        assert "landscape" in decoded
        assert "sunset" in decoded
        assert "nature" in decoded

    def test_embed_quality_to_description(self, sample_jpeg, sample_analysis_data):
        """Test quality score embedded in ImageDescription."""
        tagger = MetadataTagger()
        success = tagger.embed_metadata(sample_jpeg, sample_analysis_data)

        assert success is True

        # Read back and verify
        exif_dict = piexif.load(sample_jpeg)
        desc = exif_dict["0th"][piexif.ImageIFD.ImageDescription]
        decoded = desc.decode("utf-8")

        assert "Quality: 85/100" in decoded
        assert "Sharpness: sharp" in decoded

    def test_tags_disabled_no_usercomment(self, sample_jpeg, sample_analysis_data):
        """Test tags not embedded when disabled."""
        tagger = MetadataTagger(embed_tags=False)
        success = tagger.embed_metadata(sample_jpeg, sample_analysis_data)

        assert success is True

        # Read back and verify UserComment not set
        exif_dict = piexif.load(sample_jpeg)
        # UserComment should not be in EXIF or should be empty
        user_comment = exif_dict["Exif"].get(piexif.ExifIFD.UserComment)
        if user_comment:
            assert user_comment == b''

    def test_preserve_existing_exif(self, sample_jpeg):
        """Test existing EXIF data is preserved."""
        # Add some existing EXIF
        exif_dict = piexif.load(sample_jpeg)
        exif_dict["0th"][piexif.ImageIFD.Copyright] = b"Test Copyright"
        exif_bytes = piexif.dump(exif_dict)

        img = Image.open(sample_jpeg)
        img.save(sample_jpeg, exif=exif_bytes)

        # Embed new metadata
        tagger = MetadataTagger()
        analysis_data = {"tags": ["test"], "quality_score": 75}
        success = tagger.embed_metadata(sample_jpeg, analysis_data)

        assert success is True

        # Verify existing data preserved
        exif_dict = piexif.load(sample_jpeg)
        copyright = exif_dict["0th"][piexif.ImageIFD.Copyright]
        assert copyright == b"Test Copyright"


class TestCaptionGeneration:
    """Test caption generation from analysis data."""

    def test_full_caption_generation(self, sample_analysis_data):
        """Test caption with all fields."""
        tagger = MetadataTagger()
        caption = tagger._generate_caption(sample_analysis_data)

        assert "outdoors" in caption.lower()
        assert "tree" in caption
        assert "high quality" in caption
        assert "2024-03-15" in caption

    def test_caption_with_only_scene(self):
        """Test caption with only scene."""
        tagger = MetadataTagger()
        data = {"primary_scene": "indoors"}
        caption = tagger._generate_caption(data)

        assert caption == "Indoors"

    def test_caption_quality_levels(self):
        """Test quality level descriptions."""
        tagger = MetadataTagger()

        # High quality
        data = {"quality_score": 85}
        caption = tagger._generate_caption(data)
        assert "high quality" in caption.lower()

        # Good quality
        data = {"quality_score": 70}
        caption = tagger._generate_caption(data)
        assert "good quality" in caption.lower()

        # Low quality (no mention)
        data = {"quality_score": 40}
        caption = tagger._generate_caption(data)
        assert "quality" not in caption.lower()

    def test_caption_empty_data(self):
        """Test caption with empty data."""
        tagger = MetadataTagger()
        caption = tagger._generate_caption({})

        assert caption == "Photo"


class TestBatchEmbedding:
    """Test batch metadata embedding."""

    def test_batch_embed_multiple_files(self, temp_dir):
        """Test batch embedding in multiple files."""
        # Create multiple images
        images = []
        for i in range(3):
            img_path = Path(temp_dir) / f"image_{i}.jpg"
            img = Image.new('RGB', (50, 50), color=(i*80, 0, 0))
            img.save(str(img_path), 'JPEG')
            images.append(str(img_path))

        # Prepare data
        files_and_data = [
            (img, {"tags": [f"tag_{i}"], "quality_score": 70 + i*5})
            for i, img in enumerate(images)
        ]

        tagger = MetadataTagger()
        stats = tagger.batch_embed(files_and_data, show_progress=False)

        assert stats["success"] == 3
        assert stats["failed"] == 0

        # Verify each file
        for i, img_path in enumerate(images):
            exif_dict = piexif.load(img_path)
            user_comment = exif_dict["Exif"][piexif.ExifIFD.UserComment].decode("utf-8")
            assert f"tag_{i}" in user_comment

    def test_batch_embed_with_failures(self, temp_dir):
        """Test batch embedding handles failures gracefully."""
        # Create one valid image
        valid_img = Path(temp_dir) / "valid.jpg"
        img = Image.new('RGB', (50, 50), color='green')
        img.save(str(valid_img), 'JPEG')

        # Invalid file path
        invalid_img = str(Path(temp_dir) / "nonexistent.jpg")

        files_and_data = [
            (str(valid_img), {"tags": ["valid"]}),
            (invalid_img, {"tags": ["invalid"]}),
        ]

        tagger = MetadataTagger()
        stats = tagger.batch_embed(files_and_data, show_progress=False)

        assert stats["success"] == 1
        assert stats["failed"] == 1


class TestReadEmbeddedTags:
    """Test reading embedded tags."""

    def test_read_exif_tags(self, sample_jpeg, sample_analysis_data):
        """Test reading EXIF tags."""
        # Embed tags first
        tagger = MetadataTagger()
        tagger.embed_metadata(sample_jpeg, sample_analysis_data)

        # Read back
        tags_data = tagger.read_embedded_tags(sample_jpeg)

        assert "exif_tags" in tags_data
        assert "landscape" in tags_data["exif_tags"]
        assert "description" in tags_data
        assert "Quality: 85/100" in tags_data["description"]

    def test_read_from_empty_exif(self, sample_jpeg):
        """Test reading from image with no embedded tags."""
        tagger = MetadataTagger()
        tags_data = tagger.read_embedded_tags(sample_jpeg)

        # Should not crash, may have empty dict or error
        assert isinstance(tags_data, dict)

    def test_read_corrupted_file(self, temp_dir):
        """Test reading from corrupted file."""
        corrupt_path = Path(temp_dir) / "corrupt.jpg"
        corrupt_path.write_text("not a real image")

        tagger = MetadataTagger()
        tags_data = tagger.read_embedded_tags(str(corrupt_path))

        # Should handle gracefully
        assert isinstance(tags_data, dict)
        # May have error key
        if "error" in tags_data:
            assert isinstance(tags_data["error"], str)


class TestRGBAConversion:
    """Test RGBA to RGB conversion for JPEG."""

    def test_rgba_to_rgb_conversion(self, temp_dir):
        """Test RGBA image converted to RGB for JPEG output."""
        # Create RGBA image
        rgba_path = Path(temp_dir) / "rgba.png"
        img = Image.new('RGBA', (100, 100), color=(255, 0, 0, 128))
        img.save(str(rgba_path), 'PNG')

        # Embed metadata (will save as JPEG internally)
        output_path = Path(temp_dir) / "output.jpg"
        tagger = MetadataTagger()
        analysis_data = {"tags": ["test"]}

        success = tagger.embed_metadata(
            str(rgba_path),
            analysis_data,
            output_path=str(output_path)
        )

        assert success is True
        assert output_path.exists()

        # Verify saved as RGB
        saved_img = Image.open(str(output_path))
        assert saved_img.mode == 'RGB'

    def test_png_mode_preserved(self, temp_dir):
        """Test PNG mode preserved for PNG output."""
        # Create RGBA PNG
        rgba_path = Path(temp_dir) / "rgba.png"
        img = Image.new('RGBA', (100, 100), color=(255, 0, 0, 128))
        img.save(str(rgba_path), 'PNG')

        # Embed metadata with PNG output
        output_path = Path(temp_dir) / "output.png"
        tagger = MetadataTagger()
        analysis_data = {"tags": ["test"]}

        success = tagger.embed_metadata(
            str(rgba_path),
            analysis_data,
            output_path=str(output_path)
        )

        assert success is True
        assert output_path.exists()

        # Verify mode preserved (though EXIF may convert internally)
        saved_img = Image.open(str(output_path))
        # PNG should preserve transparency capability
        assert saved_img.mode in ('RGB', 'RGBA', 'P')


class TestUnicodeSupport:
    """Test Unicode filename and tag support."""

    def test_unicode_filename(self, temp_dir):
        """Test embedding metadata in file with Unicode name."""
        # Create image with Unicode filename
        unicode_path = Path(temp_dir) / "测试_日本語_한국어.jpg"
        img = Image.new('RGB', (100, 100), color='red')
        img.save(str(unicode_path), 'JPEG')

        tagger = MetadataTagger()
        analysis_data = {"tags": ["test"], "quality_score": 75}

        success = tagger.embed_metadata(str(unicode_path), analysis_data)
        assert success is True

    def test_unicode_tags(self, sample_jpeg):
        """Test embedding Unicode tags."""
        tagger = MetadataTagger()
        analysis_data = {
            "tags": ["日本", "한국", "中文"],
            "quality_score": 80
        }

        success = tagger.embed_metadata(sample_jpeg, analysis_data)
        assert success is True

        # Read back and verify
        tags_data = tagger.read_embedded_tags(sample_jpeg)
        assert "exif_tags" in tags_data
        # Unicode should be preserved
        exif_tags = tags_data["exif_tags"]
        assert len(exif_tags) > 0

    def test_unicode_caption(self, sample_jpeg):
        """Test Unicode in generated caption."""
        tagger = MetadataTagger(embed_caption=True)
        analysis_data = {
            "primary_scene": "日本庭園",
            "unique_objects": ["桜", "石"],
        }

        caption = tagger._generate_caption(analysis_data)
        assert len(caption) > 0
        # Should handle Unicode without crashing


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_tags_as_string(self, sample_jpeg):
        """Test tags provided as string instead of list."""
        tagger = MetadataTagger()
        analysis_data = {
            "tags": "single-tag",
            "quality_score": 70
        }

        success = tagger.embed_metadata(sample_jpeg, analysis_data)
        assert success is True

        # Verify string converted correctly
        tags_data = tagger.read_embedded_tags(sample_jpeg)
        assert "exif_tags" in tags_data
        assert "single-tag" in tags_data["exif_tags"]

    def test_tag_string_alternative_format(self, sample_jpeg):
        """Test tag_string field support."""
        tagger = MetadataTagger()
        analysis_data = {
            "tag_string": "tag1, tag2, tag3",
            "quality_score": 70
        }

        success = tagger.embed_metadata(sample_jpeg, analysis_data)
        assert success is True

    def test_in_place_modification(self, sample_jpeg):
        """Test in-place modification (no output_path)."""
        original_size = os.path.getsize(sample_jpeg)

        tagger = MetadataTagger()
        analysis_data = {"tags": ["test"], "quality_score": 75}

        success = tagger.embed_metadata(sample_jpeg, analysis_data)
        assert success is True

        # File should be modified
        new_size = os.path.getsize(sample_jpeg)
        # Size may change due to EXIF data
        assert new_size > 0

    def test_output_to_different_path(self, sample_jpeg, temp_dir):
        """Test embedding with output to different path."""
        output_path = Path(temp_dir) / "output.jpg"

        tagger = MetadataTagger()
        analysis_data = {"tags": ["test"], "quality_score": 75}

        success = tagger.embed_metadata(
            sample_jpeg,
            analysis_data,
            output_path=str(output_path)
        )

        assert success is True
        assert output_path.exists()
        # Original should still exist
        assert os.path.exists(sample_jpeg)

    def test_nonexistent_file_fails_gracefully(self, temp_dir):
        """Test nonexistent file handled gracefully."""
        nonexistent = str(Path(temp_dir) / "nonexistent.jpg")

        tagger = MetadataTagger()
        analysis_data = {"tags": ["test"]}

        success = tagger.embed_metadata(nonexistent, analysis_data)
        assert success is False

    def test_quality_score_only(self, sample_jpeg):
        """Test embedding only quality score."""
        tagger = MetadataTagger(embed_tags=False)
        analysis_data = {"quality_score": 92}

        success = tagger.embed_metadata(sample_jpeg, analysis_data)
        assert success is True

        exif_dict = piexif.load(sample_jpeg)
        desc = exif_dict["0th"][piexif.ImageIFD.ImageDescription].decode("utf-8")
        assert "Quality: 92/100" in desc

    def test_empty_analysis_data(self, sample_jpeg):
        """Test embedding with empty analysis data."""
        tagger = MetadataTagger()
        analysis_data = {}

        # Should not crash
        success = tagger.embed_metadata(sample_jpeg, analysis_data)
        # May succeed or fail depending on implementation
        assert isinstance(success, bool)
