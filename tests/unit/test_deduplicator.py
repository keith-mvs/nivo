"""Unit tests for Deduplicator with comprehensive coverage."""

import os
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import time

import pytest
from PIL import Image

from src.core.processors.deduplicator import Deduplicator


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    temp = tempfile.mkdtemp()
    yield temp
    shutil.rmtree(temp, ignore_errors=True)


@pytest.fixture
def create_test_image():
    """Factory for creating test images."""
    def _create(path, size=(100, 100), color='red', content=None):
        """Create test image with optional content."""
        if content is not None:
            # Create image with specific binary content for duplicates
            Path(path).write_bytes(content)
        else:
            img = Image.new('RGB', size, color=color)
            img.save(path, 'JPEG', quality=95)
        return path
    return _create


class TestDuplicateDetection:
    """Test duplicate file detection."""

    def test_find_exact_duplicates(self, temp_dir, create_test_image):
        """Test finding exact duplicate files."""
        # Create identical files
        content = b"test image data" * 100
        img1 = str(Path(temp_dir) / "image1.jpg")
        img2 = str(Path(temp_dir) / "image2.jpg")
        img3 = str(Path(temp_dir) / "unique.jpg")

        create_test_image(img1, content=content)
        create_test_image(img2, content=content)
        create_test_image(img3, content=content + b"different")

        deduplicator = Deduplicator()
        duplicates = deduplicator.find_duplicates(
            [img1, img2, img3],
            show_progress=False
        )

        # Should find one duplicate set (img1 and img2)
        assert len(duplicates) == 1
        duplicate_set = list(duplicates.values())[0]
        assert len(duplicate_set) == 2
        assert img1 in duplicate_set
        assert img2 in duplicate_set
        assert img3 not in duplicate_set

    def test_no_duplicates_found(self, temp_dir, create_test_image):
        """Test when no duplicates exist."""
        # Create unique files
        img1 = str(Path(temp_dir) / "image1.jpg")
        img2 = str(Path(temp_dir) / "image2.jpg")

        create_test_image(img1, content=b"unique1" * 100)
        create_test_image(img2, content=b"unique2" * 100)

        deduplicator = Deduplicator()
        duplicates = deduplicator.find_duplicates(
            [img1, img2],
            show_progress=False
        )

        assert len(duplicates) == 0

    def test_multiple_duplicate_sets(self, temp_dir, create_test_image):
        """Test finding multiple sets of duplicates."""
        # Create two sets of duplicates
        content_a = b"content_a" * 100
        content_b = b"content_b" * 100

        img1a = str(Path(temp_dir) / "set_a_1.jpg")
        img2a = str(Path(temp_dir) / "set_a_2.jpg")
        img1b = str(Path(temp_dir) / "set_b_1.jpg")
        img2b = str(Path(temp_dir) / "set_b_2.jpg")

        create_test_image(img1a, content=content_a)
        create_test_image(img2a, content=content_a)
        create_test_image(img1b, content=content_b)
        create_test_image(img2b, content=content_b)

        deduplicator = Deduplicator()
        duplicates = deduplicator.find_duplicates(
            [img1a, img2a, img1b, img2b],
            show_progress=False
        )

        # Should find two duplicate sets
        assert len(duplicates) == 2
        for duplicate_set in duplicates.values():
            assert len(duplicate_set) == 2


class TestQuickHashScreening:
    """Test quick hash screening optimization."""

    def test_quick_hash_reduces_candidates(self, temp_dir, create_test_image):
        """Test quick hash screening reduces candidate set."""
        # Create many unique files and a few duplicates
        files = []

        # Create 150 unique files
        for i in range(150):
            path = str(Path(temp_dir) / f"unique_{i}.jpg")
            create_test_image(path, content=f"unique_{i}".encode() * 100)
            files.append(path)

        # Create duplicates
        dup_content = b"duplicate" * 100
        dup1 = str(Path(temp_dir) / "dup1.jpg")
        dup2 = str(Path(temp_dir) / "dup2.jpg")
        create_test_image(dup1, content=dup_content)
        create_test_image(dup2, content=dup_content)
        files.extend([dup1, dup2])

        # Enable quick hash (should screen out unique files)
        deduplicator = Deduplicator(use_quick_hash=True)
        duplicates = deduplicator.find_duplicates(files, show_progress=False)

        # Should find the duplicate pair
        assert len(duplicates) == 1
        duplicate_set = list(duplicates.values())[0]
        assert dup1 in duplicate_set
        assert dup2 in duplicate_set

    def test_quick_hash_disabled(self, temp_dir, create_test_image):
        """Test quick hash can be disabled."""
        content = b"test" * 100
        img1 = str(Path(temp_dir) / "img1.jpg")
        img2 = str(Path(temp_dir) / "img2.jpg")

        create_test_image(img1, content=content)
        create_test_image(img2, content=content)

        # Disable quick hash
        deduplicator = Deduplicator(use_quick_hash=False)
        duplicates = deduplicator.find_duplicates(
            [img1, img2],
            show_progress=False
        )

        # Should still find duplicates
        assert len(duplicates) == 1


class TestKeepStrategies:
    """Test file selection strategies."""

    def test_keep_highest_quality(self, temp_dir, create_test_image):
        """Test highest quality strategy selects largest/highest res."""
        # Create duplicates with different resolutions
        img1 = str(Path(temp_dir) / "small.jpg")
        img2 = str(Path(temp_dir) / "large.jpg")

        create_test_image(img1, size=(50, 50), color='red')
        create_test_image(img2, size=(200, 200), color='red')

        # Make them "duplicates" for testing (copy content)
        deduplicator = Deduplicator()
        best = deduplicator.select_best_to_keep(
            [img1, img2],
            strategy="highest_quality"
        )

        # Should keep larger resolution
        assert best == img2

    def test_keep_oldest(self, temp_dir, create_test_image):
        """Test oldest strategy selects oldest modified time."""
        img1 = str(Path(temp_dir) / "old.jpg")
        img2 = str(Path(temp_dir) / "new.jpg")

        create_test_image(img1, content=b"test" * 100)
        time.sleep(0.1)  # Ensure different mtime
        create_test_image(img2, content=b"test" * 100)

        deduplicator = Deduplicator()
        best = deduplicator.select_best_to_keep(
            [img1, img2],
            strategy="oldest"
        )

        # Should keep older file
        assert best == img1

    def test_keep_newest(self, temp_dir, create_test_image):
        """Test newest strategy selects newest modified time."""
        img1 = str(Path(temp_dir) / "old.jpg")
        img2 = str(Path(temp_dir) / "new.jpg")

        create_test_image(img1, content=b"test" * 100)
        time.sleep(0.1)  # Ensure different mtime
        create_test_image(img2, content=b"test" * 100)

        deduplicator = Deduplicator()
        best = deduplicator.select_best_to_keep(
            [img1, img2],
            strategy="newest"
        )

        # Should keep newer file
        assert best == img2

    def test_keep_largest(self, temp_dir, create_test_image):
        """Test largest strategy selects largest file size."""
        img1 = str(Path(temp_dir) / "small.jpg")
        img2 = str(Path(temp_dir) / "large.jpg")

        # Create files with different sizes
        create_test_image(img1, content=b"small" * 10)
        create_test_image(img2, content=b"large" * 1000)

        deduplicator = Deduplicator()
        best = deduplicator.select_best_to_keep(
            [img1, img2],
            strategy="largest"
        )

        # Should keep larger file
        assert best == img2

    def test_unknown_strategy_fallback(self, temp_dir, create_test_image):
        """Test unknown strategy falls back to first file."""
        img1 = str(Path(temp_dir) / "img1.jpg")
        img2 = str(Path(temp_dir) / "img2.jpg")

        create_test_image(img1, content=b"test")
        create_test_image(img2, content=b"test")

        deduplicator = Deduplicator()
        best = deduplicator.select_best_to_keep(
            [img1, img2],
            strategy="unknown_strategy"
        )

        # Should fall back to first
        assert best == img1


class TestDeletionWithDryRun:
    """Test file deletion with dry run mode."""

    def test_dry_run_no_deletion(self, temp_dir, create_test_image):
        """Test dry run doesn't delete files."""
        content = b"duplicate" * 100
        img1 = str(Path(temp_dir) / "img1.jpg")
        img2 = str(Path(temp_dir) / "img2.jpg")

        create_test_image(img1, content=content)
        create_test_image(img2, content=content)

        deduplicator = Deduplicator()
        duplicates = deduplicator.find_duplicates(
            [img1, img2],
            show_progress=False
        )

        stats = deduplicator.remove_duplicates(
            duplicates,
            strategy="newest",
            dry_run=True
        )

        # Files should still exist
        assert os.path.exists(img1)
        assert os.path.exists(img2)
        # Stats should show what would be deleted
        assert len(stats["files_to_delete"]) == 1
        assert stats["dry_run"] is True

    def test_execute_mode_deletes_files(self, temp_dir, create_test_image):
        """Test execute mode actually deletes files."""
        content = b"duplicate" * 100
        img1 = str(Path(temp_dir) / "img1.jpg")
        img2 = str(Path(temp_dir) / "img2.jpg")

        create_test_image(img1, content=content)
        time.sleep(0.1)  # Ensure different mtime
        create_test_image(img2, content=content)

        deduplicator = Deduplicator()
        duplicates = deduplicator.find_duplicates(
            [img1, img2],
            show_progress=False
        )

        stats = deduplicator.remove_duplicates(
            duplicates,
            strategy="newest",
            dry_run=False
        )

        # Newer file kept, older deleted
        assert not os.path.exists(img1)
        assert os.path.exists(img2)
        assert len(stats["files_to_delete"]) == 1
        assert img1 in stats["files_to_delete"]
        assert img2 in stats["files_to_keep"]

    def test_space_saved_calculation(self, temp_dir, create_test_image):
        """Test space saved calculation."""
        content = b"duplicate" * 1000
        img1 = str(Path(temp_dir) / "img1.jpg")
        img2 = str(Path(temp_dir) / "img2.jpg")

        create_test_image(img1, content=content)
        create_test_image(img2, content=content)

        file_size = os.path.getsize(img1)

        deduplicator = Deduplicator()
        duplicates = deduplicator.find_duplicates(
            [img1, img2],
            show_progress=False
        )

        stats = deduplicator.remove_duplicates(
            duplicates,
            dry_run=True
        )

        # Space saved should match deleted file size
        assert stats["space_saved"] == file_size


class TestReportGeneration:
    """Test duplicate report generation."""

    def test_generate_report(self, temp_dir, create_test_image):
        """Test report generation with duplicates."""
        content = b"duplicate" * 100
        img1 = str(Path(temp_dir) / "img1.jpg")
        img2 = str(Path(temp_dir) / "img2.jpg")

        create_test_image(img1, content=content)
        create_test_image(img2, content=content)

        deduplicator = Deduplicator()
        duplicates = deduplicator.find_duplicates(
            [img1, img2],
            show_progress=False
        )

        report = deduplicator.generate_report(duplicates)

        # Verify report content
        assert "Duplicate Files Report" in report
        assert "Total duplicate sets: 1" in report
        assert img1 in report
        assert img2 in report

    def test_save_report_to_file(self, temp_dir, create_test_image):
        """Test saving report to file."""
        content = b"duplicate" * 100
        img1 = str(Path(temp_dir) / "img1.jpg")
        img2 = str(Path(temp_dir) / "img2.jpg")

        create_test_image(img1, content=content)
        create_test_image(img2, content=content)

        deduplicator = Deduplicator()
        duplicates = deduplicator.find_duplicates(
            [img1, img2],
            show_progress=False
        )

        report_path = str(Path(temp_dir) / "report.md")
        report = deduplicator.generate_report(duplicates, output_path=report_path)

        # Verify report file created
        assert os.path.exists(report_path)

        # Verify content matches
        with open(report_path, 'r', encoding='utf-8') as f:
            saved_report = f.read()
        assert saved_report == report

    def test_report_with_multiple_sets(self, temp_dir, create_test_image):
        """Test report with multiple duplicate sets."""
        content_a = b"set_a" * 100
        content_b = b"set_b" * 100

        img1a = str(Path(temp_dir) / "a1.jpg")
        img2a = str(Path(temp_dir) / "a2.jpg")
        img1b = str(Path(temp_dir) / "b1.jpg")
        img2b = str(Path(temp_dir) / "b2.jpg")

        create_test_image(img1a, content=content_a)
        create_test_image(img2a, content=content_a)
        create_test_image(img1b, content=content_b)
        create_test_image(img2b, content=content_b)

        deduplicator = Deduplicator()
        duplicates = deduplicator.find_duplicates(
            [img1a, img2a, img1b, img2b],
            show_progress=False
        )

        report = deduplicator.generate_report(duplicates)

        # Should mention both sets
        assert "Total duplicate sets: 2" in report
        assert "Duplicate Set 1" in report
        assert "Duplicate Set 2" in report


class TestHashAlgorithms:
    """Test different hash algorithms."""

    def test_sha256_algorithm(self, temp_dir, create_test_image):
        """Test SHA256 hash algorithm."""
        content = b"test" * 100
        img1 = str(Path(temp_dir) / "img1.jpg")
        img2 = str(Path(temp_dir) / "img2.jpg")

        create_test_image(img1, content=content)
        create_test_image(img2, content=content)

        deduplicator = Deduplicator(hash_algorithm="sha256")
        duplicates = deduplicator.find_duplicates(
            [img1, img2],
            show_progress=False
        )

        assert len(duplicates) == 1

    def test_md5_algorithm(self, temp_dir, create_test_image):
        """Test MD5 hash algorithm."""
        content = b"test" * 100
        img1 = str(Path(temp_dir) / "img1.jpg")
        img2 = str(Path(temp_dir) / "img2.jpg")

        create_test_image(img1, content=content)
        create_test_image(img2, content=content)

        deduplicator = Deduplicator(hash_algorithm="md5")
        duplicates = deduplicator.find_duplicates(
            [img1, img2],
            show_progress=False
        )

        assert len(duplicates) == 1


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_nonexistent_file_handled(self, temp_dir, create_test_image):
        """Test nonexistent files handled gracefully."""
        img1 = str(Path(temp_dir) / "exists.jpg")
        img2 = str(Path(temp_dir) / "nonexistent.jpg")

        create_test_image(img1, content=b"test")
        # img2 doesn't exist

        deduplicator = Deduplicator()
        # Should not crash
        duplicates = deduplicator.find_duplicates(
            [img1, img2],
            show_progress=False
        )

        # May or may not find duplicates, but shouldn't crash
        assert isinstance(duplicates, dict)

    def test_empty_file_list(self):
        """Test empty file list."""
        deduplicator = Deduplicator()
        duplicates = deduplicator.find_duplicates([], show_progress=False)

        assert len(duplicates) == 0

    def test_single_file(self, temp_dir, create_test_image):
        """Test single file (no duplicates possible)."""
        img1 = str(Path(temp_dir) / "single.jpg")
        create_test_image(img1, content=b"test")

        deduplicator = Deduplicator()
        duplicates = deduplicator.find_duplicates([img1], show_progress=False)

        assert len(duplicates) == 0

    def test_parallel_workers_configuration(self, temp_dir, create_test_image):
        """Test configurable parallel workers."""
        content = b"test" * 100
        img1 = str(Path(temp_dir) / "img1.jpg")
        img2 = str(Path(temp_dir) / "img2.jpg")

        create_test_image(img1, content=content)
        create_test_image(img2, content=content)

        # Test with different worker counts
        for workers in [1, 2, 8]:
            deduplicator = Deduplicator(max_workers=workers)
            duplicates = deduplicator.find_duplicates(
                [img1, img2],
                show_progress=False
            )

            assert len(duplicates) == 1
