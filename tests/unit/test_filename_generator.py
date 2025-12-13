"""Tests for filename generator utility."""

import pytest
from datetime import datetime
from pathlib import Path

from src.core.utils.filename_generator import (
    FilenameGenerator,
    generate_unique_filename,
    validate_filename,
)


class TestFilenameGenerator:
    """Tests for FilenameGenerator class."""

    def test_generate_default(self):
        """Generate filename with default settings."""
        gen = FilenameGenerator()
        filename = gen.generate()

        assert filename.startswith("img_")
        assert filename.endswith(".png")
        assert len(filename) == 32  # img_(4) + timestamp(15) + _(1) + uuid(8) + .png(4)

    def test_generate_with_extension(self):
        """Generate filename with custom extension."""
        gen = FilenameGenerator()

        for ext in ["jpg", "jpeg", "webp", "tiff"]:
            filename = gen.generate(extension=ext)
            assert filename.endswith(f".{ext}")

    def test_generate_with_timestamp(self):
        """Generate filename with specific timestamp."""
        gen = FilenameGenerator()
        ts = datetime(2025, 12, 13, 14, 30, 52)
        filename = gen.generate(timestamp=ts)

        assert "20251213_143052" in filename

    def test_generate_invalid_extension_raises(self):
        """Invalid extension raises ValueError."""
        gen = FilenameGenerator()

        with pytest.raises(ValueError, match="Invalid extension"):
            gen.generate(extension="exe")

    def test_generate_unique_filenames(self):
        """Each generated filename is unique."""
        gen = FilenameGenerator()
        filenames = {gen.generate() for _ in range(100)}

        assert len(filenames) == 100

    def test_generate_path(self):
        """Generate full path with directory."""
        gen = FilenameGenerator()
        path = gen.generate_path("C:/output")

        assert isinstance(path, Path)
        assert str(path).startswith("C:")
        assert path.name.startswith("img_")

    def test_custom_prefix(self):
        """Custom prefix in filename."""
        gen = FilenameGenerator(prefix="processed")
        filename = gen.generate()

        assert filename.startswith("processed_")

    def test_custom_uuid_length(self):
        """Custom UUID length."""
        gen = FilenameGenerator(uuid_length=12)
        filename = gen.generate()

        # Extract UUID part: after timestamp, before extension
        parts = filename.replace(".png", "").split("_")
        uuid_part = parts[-1]

        assert len(uuid_part) == 12


class TestFilenameValidation:
    """Tests for filename validation."""

    def test_validate_valid_filename(self):
        """Valid filename passes validation."""
        is_valid, error = FilenameGenerator.validate("img_20251213_143052_a1b2c3d4.png")

        assert is_valid is True
        assert error is None

    def test_validate_valid_with_path(self):
        """Valid filename with path passes validation."""
        is_valid, error = FilenameGenerator.validate(
            "C:/output/img_20251213_143052_a1b2c3d4.jpg"
        )

        assert is_valid is True
        assert error is None

    def test_validate_invalid_prefix(self):
        """Invalid prefix fails validation."""
        is_valid, error = FilenameGenerator.validate("photo_20251213_143052_a1b2c3d4.png")

        assert is_valid is False
        assert "does not match pattern" in error

    def test_validate_invalid_timestamp(self):
        """Invalid timestamp format fails validation."""
        is_valid, error = FilenameGenerator.validate("img_2025-12-13_143052_a1b2c3d4.png")

        assert is_valid is False
        assert "does not match pattern" in error

    def test_validate_invalid_extension(self):
        """Invalid extension fails validation."""
        is_valid, error = FilenameGenerator.validate("img_20251213_143052_a1b2c3d4.exe")

        assert is_valid is False
        assert "Invalid extension" in error or "does not match pattern" in error

    def test_validate_before_save_valid(self):
        """validate_before_save passes for valid filename."""
        # Should not raise
        FilenameGenerator.validate_before_save("img_20251213_143052_a1b2c3d4.png")

    def test_validate_before_save_invalid_raises(self):
        """validate_before_save raises for invalid filename."""
        with pytest.raises(ValueError, match="Cannot save"):
            FilenameGenerator.validate_before_save("invalid_filename.png")


class TestFilenameParsing:
    """Tests for filename parsing."""

    def test_parse_valid_filename(self):
        """Parse valid filename into components."""
        result = FilenameGenerator.parse("img_20251213_143052_a1b2c3d4.png")

        assert result is not None
        assert result["timestamp"] == datetime(2025, 12, 13, 14, 30, 52)
        assert result["uuid"] == "a1b2c3d4"
        assert result["extension"] == "png"

    def test_parse_invalid_filename(self):
        """Parse returns None for invalid filename."""
        result = FilenameGenerator.parse("random_file.jpg")

        assert result is None


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_generate_unique_filename(self):
        """generate_unique_filename returns valid filename."""
        filename = generate_unique_filename()

        assert validate_filename(filename) is True

    def test_generate_unique_filename_with_dir(self):
        """generate_unique_filename with output directory."""
        path = generate_unique_filename(output_dir="C:/output")

        assert "C:" in path
        assert "output" in path

    def test_validate_filename_convenience(self):
        """validate_filename convenience function."""
        assert validate_filename("img_20251213_143052_a1b2c3d4.png") is True
        assert validate_filename("invalid.png") is False


class TestRoundTrip:
    """Test generation and validation together."""

    def test_generated_filenames_are_valid(self):
        """All generated filenames pass validation."""
        gen = FilenameGenerator()

        for _ in range(50):
            filename = gen.generate()
            is_valid, error = FilenameGenerator.validate(filename)

            assert is_valid is True, f"Generated filename failed: {filename}, {error}"

    def test_generated_paths_are_valid(self):
        """All generated paths pass validation."""
        gen = FilenameGenerator()

        for ext in ["png", "jpg", "webp"]:
            path = gen.generate_path("C:/output", extension=ext)
            is_valid, error = FilenameGenerator.validate(str(path))

            assert is_valid is True, f"Generated path failed: {path}, {error}"
