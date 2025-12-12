"""Unit tests for ImageRenamer with comprehensive coverage."""

import os
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict

import pytest

from src.core.processors.renamer import ImageRenamer


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    temp = tempfile.mkdtemp()
    yield temp
    shutil.rmtree(temp, ignore_errors=True)


@pytest.fixture
def sample_image(temp_dir):
    """Create a sample image file."""
    image_path = Path(temp_dir) / "test_image.jpg"
    image_path.write_text("fake image data")
    return str(image_path)


@pytest.fixture
def sample_metadata():
    """Create sample metadata for testing."""
    return {
        "file_path": "test_image.jpg",
        "datetime_original": datetime(2024, 3, 15, 14, 30, 45),
        "model": "Canon EOS 5D",
        "tags": ["landscape", "sunset", "nature"],
    }


class TestPatternSubstitution:
    """Test filename pattern substitution."""

    def test_datetime_pattern(self, sample_metadata):
        """Test {datetime} pattern substitution."""
        renamer = ImageRenamer(pattern="{datetime}")
        name = renamer._generate_name(sample_metadata, set())
        assert name == "2024-03-15_143045.jpg"

    def test_date_pattern(self, sample_metadata):
        """Test {date} pattern substitution."""
        renamer = ImageRenamer(pattern="{date}")
        name = renamer._generate_name(sample_metadata, set())
        assert name == "2024-03-15.jpg"

    def test_time_pattern(self, sample_metadata):
        """Test {time} pattern substitution."""
        renamer = ImageRenamer(pattern="{time}")
        name = renamer._generate_name(sample_metadata, set())
        assert name == "143045.jpg"

    def test_year_month_day_pattern(self, sample_metadata):
        """Test {year}, {month}, {day} patterns."""
        renamer = ImageRenamer(pattern="{year}_{month}_{day}")
        name = renamer._generate_name(sample_metadata, set())
        assert name == "2024_03_15.jpg"

    def test_camera_pattern(self, sample_metadata):
        """Test {camera} pattern substitution."""
        renamer = ImageRenamer(pattern="{camera}_{datetime}")
        name = renamer._generate_name(sample_metadata, set())
        assert name == "Canon-EOS-5D_2024-03-15_143045.jpg"

    def test_tags_pattern(self, sample_metadata):
        """Test {tags} pattern substitution."""
        renamer = ImageRenamer(pattern="{datetime}_{tags}")
        name = renamer._generate_name(sample_metadata, set())
        assert name == "2024-03-15_143045_landscape_sunset_nature.jpg"

    def test_custom_date_format(self, sample_metadata):
        """Test custom date format."""
        renamer = ImageRenamer(
            pattern="{date}",
            date_format="%d-%b-%Y"
        )
        name = renamer._generate_name(sample_metadata, set())
        assert name == "15-Mar-2024.jpg"

    def test_missing_datetime_fallback(self):
        """Test fallback to current time when datetime missing."""
        metadata = {"file_path": "test.jpg"}
        renamer = ImageRenamer(pattern="{datetime}")
        name = renamer._generate_name(metadata, set())
        # Should use current time (just verify it has expected format)
        assert len(name) > 10
        assert name.endswith(".jpg")


class TestCollisionHandling:
    """Test filename collision handling."""

    def test_no_collision(self, sample_metadata, temp_dir):
        """Test when no collision exists."""
        renamer = ImageRenamer(pattern="{datetime}")
        path = str(Path(temp_dir) / "2024-03-15_143045.jpg")
        result = renamer._handle_collision(path, set())
        assert result == path

    def test_collision_adds_sequence(self, sample_metadata, temp_dir):
        """Test collision adds sequence number."""
        renamer = ImageRenamer(pattern="{datetime}")
        base_name = "2024-03-15_143045.jpg"
        path = str(Path(temp_dir) / base_name)

        # Simulate existing name
        used_names = {base_name}
        result = renamer._handle_collision(path, used_names)

        assert result.endswith("_001.jpg")

    def test_multiple_collisions(self, temp_dir):
        """Test multiple collisions increment sequence."""
        renamer = ImageRenamer(pattern="{datetime}")
        base_name = "2024-03-15_143045.jpg"
        path = str(Path(temp_dir) / base_name)

        used_names = {
            "2024-03-15_143045.jpg",
            "2024-03-15_143045_001.jpg",
            "2024-03-15_143045_002.jpg",
        }
        result = renamer._handle_collision(path, used_names)

        assert result.endswith("_003.jpg")

    def test_custom_collision_suffix(self, temp_dir):
        """Test custom collision suffix format."""
        renamer = ImageRenamer(
            pattern="{datetime}",
            collision_suffix="-{seq:02d}"
        )
        base_name = "2024-03-15_143045.jpg"
        path = str(Path(temp_dir) / base_name)
        used_names = {base_name}

        result = renamer._handle_collision(path, used_names)
        assert result.endswith("-01.jpg")


class TestFilenameSanitization:
    """Test filename sanitization."""

    def test_remove_invalid_windows_chars(self):
        """Test removal of invalid Windows characters."""
        renamer = ImageRenamer()

        # Test all invalid Windows chars
        dirty = 'test<>:"/\\|?*file'
        clean = renamer._sanitize_filename(dirty)
        assert clean == "testfile"

    def test_replace_spaces_with_underscores(self):
        """Test spaces replaced with underscores."""
        renamer = ImageRenamer()
        dirty = "my vacation photo"
        clean = renamer._sanitize_filename(dirty)
        assert clean == "my_vacation_photo"

    def test_remove_consecutive_underscores(self):
        """Test consecutive underscores removed."""
        renamer = ImageRenamer()
        dirty = "test___multiple____underscores"
        clean = renamer._sanitize_filename(dirty)
        assert clean == "test_multiple_underscores"

    def test_strip_leading_trailing_underscores(self):
        """Test leading/trailing underscores removed."""
        renamer = ImageRenamer()
        dirty = "_test_file_"
        clean = renamer._sanitize_filename(dirty)
        assert clean == "test_file"


class TestLengthEnforcement:
    """Test filename length enforcement."""

    def test_long_filename_truncated(self, sample_metadata):
        """Test long filenames are truncated."""
        # Create very long tag list
        sample_metadata["tags"] = ["tag" + str(i) for i in range(50)]

        renamer = ImageRenamer(
            pattern="{datetime}_{tags}",
            max_filename_length=100
        )
        name = renamer._generate_name(sample_metadata, set())

        assert len(name) <= 100

    def test_extension_preserved_after_truncation(self, sample_metadata):
        """Test file extension preserved after truncation."""
        sample_metadata["tags"] = ["verylongtag" * 20]

        renamer = ImageRenamer(
            pattern="{tags}",
            max_filename_length=50
        )
        name = renamer._generate_name(sample_metadata, set())

        assert name.endswith(".jpg")
        assert len(name) <= 50


class TestDryRunMode:
    """Test dry run vs execute mode."""

    def test_dry_run_no_changes(self, temp_dir, sample_image):
        """Test dry run doesn't modify files."""
        metadata = [{"file_path": sample_image}]
        renamer = ImageRenamer(pattern="{datetime}")

        rename_map = renamer.rename_files(metadata, dry_run=True)

        # Original file should still exist
        assert os.path.exists(sample_image)
        # New file should NOT exist
        new_path = list(rename_map.values())[0]
        assert not os.path.exists(new_path)

    def test_execute_mode_renames_files(self, temp_dir, sample_image):
        """Test execute mode actually renames files."""
        metadata = [{
            "file_path": sample_image,
            "datetime_original": datetime(2024, 3, 15, 14, 30, 45)
        }]
        renamer = ImageRenamer(
            pattern="{datetime}",
            preserve_original=False
        )

        rename_map = renamer.rename_files(metadata, dry_run=False)
        new_path = list(rename_map.values())[0]

        # Original should be gone (no preserve)
        assert not os.path.exists(sample_image)
        # New file should exist
        assert os.path.exists(new_path)


class TestBackupCreation:
    """Test backup file creation."""

    def test_preserve_original_creates_backup(self, temp_dir, sample_image):
        """Test preserve_original creates backup file."""
        metadata = [{
            "file_path": sample_image,
            "datetime_original": datetime(2024, 3, 15, 14, 30, 45)
        }]
        renamer = ImageRenamer(
            pattern="{datetime}",
            preserve_original=True
        )

        rename_map = renamer.rename_files(metadata, dry_run=False)

        # Backup should exist
        backup_path = f"{sample_image}.backup"
        assert os.path.exists(backup_path)

    def test_no_preserve_no_backup(self, temp_dir, sample_image):
        """Test preserve_original=False doesn't create backup."""
        metadata = [{
            "file_path": sample_image,
            "datetime_original": datetime(2024, 3, 15, 14, 30, 45)
        }]
        renamer = ImageRenamer(
            pattern="{datetime}",
            preserve_original=False
        )

        rename_map = renamer.rename_files(metadata, dry_run=False)

        # Backup should NOT exist
        backup_path = f"{sample_image}.backup"
        assert not os.path.exists(backup_path)

    def test_output_dir_no_backup(self, temp_dir, sample_image):
        """Test output_dir mode doesn't create backup (copies instead)."""
        output_dir = os.path.join(temp_dir, "output")
        metadata = [{
            "file_path": sample_image,
            "datetime_original": datetime(2024, 3, 15, 14, 30, 45)
        }]
        renamer = ImageRenamer(
            pattern="{datetime}",
            preserve_original=True  # Should be ignored for output_dir
        )

        rename_map = renamer.rename_files(
            metadata,
            output_dir=output_dir,
            dry_run=False
        )

        # Original should still exist (copied, not moved)
        assert os.path.exists(sample_image)
        # No backup created
        backup_path = f"{sample_image}.backup"
        assert not os.path.exists(backup_path)
        # New file in output dir
        new_path = list(rename_map.values())[0]
        assert os.path.exists(new_path)
        assert output_dir in new_path


class TestUnicodeFilenames:
    """Test Windows Unicode filename support."""

    def test_unicode_source_filename(self, temp_dir):
        """Test renaming files with Unicode characters in source name."""
        # Create file with Unicode name
        unicode_path = Path(temp_dir) / "测试图片_日本語_한국어.jpg"
        unicode_path.write_text("fake image data")

        metadata = [{
            "file_path": str(unicode_path),
            "datetime_original": datetime(2024, 3, 15, 14, 30, 45)
        }]
        renamer = ImageRenamer(
            pattern="{datetime}",
            preserve_original=False
        )

        rename_map = renamer.rename_files(metadata, dry_run=False)

        # Should successfully rename
        assert len(rename_map) == 1
        new_path = list(rename_map.values())[0]
        assert os.path.exists(new_path)

    def test_unicode_tag_sanitization(self):
        """Test Unicode characters in tags are handled."""
        metadata = {
            "file_path": "test.jpg",
            "datetime_original": datetime(2024, 3, 15, 14, 30, 45),
            "tags": ["日本", "한국", "中文"]
        }
        renamer = ImageRenamer(pattern="{datetime}_{tags}")

        # Should not crash
        name = renamer._generate_name(metadata, set())
        assert name.endswith(".jpg")


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_missing_file_path_skipped(self):
        """Test metadata without file_path is skipped."""
        metadata = [{"datetime_original": datetime.now()}]
        renamer = ImageRenamer(pattern="{datetime}")

        rename_map = renamer.rename_files(metadata, dry_run=True)
        assert len(rename_map) == 0

    def test_nonexistent_file_skipped(self, temp_dir):
        """Test nonexistent file is skipped."""
        metadata = [{
            "file_path": os.path.join(temp_dir, "nonexistent.jpg"),
            "datetime_original": datetime.now()
        }]
        renamer = ImageRenamer(pattern="{datetime}")

        rename_map = renamer.rename_files(metadata, dry_run=True)
        assert len(rename_map) == 0

    def test_datetime_string_parsing(self):
        """Test datetime as string is parsed correctly."""
        metadata = {
            "file_path": "test.jpg",
            "datetime_original": "2024:03:15 14:30:45"  # EXIF format
        }
        renamer = ImageRenamer(pattern="{datetime}")
        name = renamer._generate_name(metadata, set())

        assert "2024-03-15_143045" in name

    def test_batch_renaming_preserves_order(self, temp_dir):
        """Test batch renaming maintains consistent naming."""
        # Create multiple test images
        images = []
        for i in range(5):
            path = Path(temp_dir) / f"image_{i}.jpg"
            path.write_text(f"image {i}")
            images.append(str(path))

        metadata = [
            {
                "file_path": img,
                "datetime_original": datetime(2024, 3, 15, 14, 30, i)
            }
            for i, img in enumerate(images)
        ]

        renamer = ImageRenamer(
            pattern="{datetime}",
            preserve_original=False
        )
        rename_map = renamer.rename_files(metadata, dry_run=False)

        # All files should be renamed
        assert len(rename_map) == 5
        # All new files should exist
        for new_path in rename_map.values():
            assert os.path.exists(new_path)
