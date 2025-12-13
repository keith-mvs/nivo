"""Unique filename generation for processed images with validation."""

import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple


class FilenameGenerator:
    """Generate and validate unique filenames for processed images.

    Standard scheme: img_{timestamp}_{uuid}.{ext}
    Example: img_20251213_143052_a1b2c3d4.png
    """

    # Regex pattern for validation
    PATTERN = re.compile(
        r'^img_'                           # Prefix
        r'(\d{8}_\d{6})'                   # Timestamp: YYYYMMDD_HHMMSS
        r'_([a-f0-9]{8})'                  # UUID (first 8 chars)
        r'\.([a-z]+)$',                    # Extension
        re.IGNORECASE
    )

    VALID_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'tiff', 'bmp', 'gif'}

    def __init__(
        self,
        prefix: str = "img",
        timestamp_format: str = "%Y%m%d_%H%M%S",
        uuid_length: int = 8,
    ):
        """Initialize filename generator.

        Args:
            prefix: Filename prefix (default: "img")
            timestamp_format: strftime format for timestamp
            uuid_length: Number of UUID hex characters to use (4-32)
        """
        self.prefix = prefix
        self.timestamp_format = timestamp_format
        self.uuid_length = max(4, min(32, uuid_length))

    def generate(
        self,
        extension: str = "png",
        timestamp: Optional[datetime] = None,
    ) -> str:
        """Generate a unique filename.

        Args:
            extension: File extension without dot (default: "png")
            timestamp: Optional datetime, uses current time if None

        Returns:
            Unique filename like "img_20251213_143052_a1b2c3d4.png"

        Raises:
            ValueError: If extension is invalid
        """
        ext = extension.lower().lstrip('.')
        if ext not in self.VALID_EXTENSIONS:
            raise ValueError(
                f"Invalid extension '{ext}'. "
                f"Valid: {', '.join(sorted(self.VALID_EXTENSIONS))}"
            )

        ts = timestamp or datetime.now()
        ts_str = ts.strftime(self.timestamp_format)
        uid = uuid.uuid4().hex[:self.uuid_length]

        return f"{self.prefix}_{ts_str}_{uid}.{ext}"

    def generate_path(
        self,
        output_dir: str,
        extension: str = "png",
        timestamp: Optional[datetime] = None,
    ) -> Path:
        """Generate a unique file path.

        Args:
            output_dir: Directory for the output file
            extension: File extension without dot
            timestamp: Optional datetime

        Returns:
            Full Path object for the new file
        """
        filename = self.generate(extension=extension, timestamp=timestamp)
        return Path(output_dir) / filename

    @classmethod
    def validate(cls, filename: str) -> Tuple[bool, Optional[str]]:
        """Validate filename matches the standard naming convention.

        Args:
            filename: Filename to validate (with or without path)

        Returns:
            Tuple of (is_valid, error_message)
            error_message is None if valid
        """
        name = Path(filename).name

        match = cls.PATTERN.match(name)
        if not match:
            return False, (
                f"Filename '{name}' does not match pattern "
                "'img_YYYYMMDD_HHMMSS_uuid.ext'"
            )

        timestamp_str, uid, ext = match.groups()

        # Validate timestamp is a real date
        try:
            datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
        except ValueError:
            return False, f"Invalid timestamp '{timestamp_str}'"

        # Validate extension
        if ext.lower() not in cls.VALID_EXTENSIONS:
            return False, (
                f"Invalid extension '{ext}'. "
                f"Valid: {', '.join(sorted(cls.VALID_EXTENSIONS))}"
            )

        return True, None

    @classmethod
    def validate_before_save(cls, filepath: str) -> None:
        """Validate filename before saving, raise if invalid.

        Args:
            filepath: Full path or filename to validate

        Raises:
            ValueError: If filename doesn't match convention
        """
        is_valid, error = cls.validate(filepath)
        if not is_valid:
            raise ValueError(f"Cannot save: {error}")

    @classmethod
    def parse(cls, filename: str) -> Optional[dict]:
        """Parse a valid filename into components.

        Args:
            filename: Filename to parse

        Returns:
            Dict with 'timestamp', 'uuid', 'extension' keys, or None if invalid
        """
        name = Path(filename).name
        match = cls.PATTERN.match(name)

        if not match:
            return None

        timestamp_str, uid, ext = match.groups()

        return {
            "timestamp": datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S"),
            "uuid": uid,
            "extension": ext.lower(),
        }


def generate_unique_filename(
    extension: str = "png",
    output_dir: Optional[str] = None,
) -> str:
    """Convenience function to generate a unique filename.

    Args:
        extension: File extension without dot
        output_dir: Optional directory to prepend

    Returns:
        Unique filename or full path if output_dir provided
    """
    generator = FilenameGenerator()

    if output_dir:
        return str(generator.generate_path(output_dir, extension))
    return generator.generate(extension)


def validate_filename(filepath: str) -> bool:
    """Convenience function to validate a filename.

    Args:
        filepath: Path or filename to validate

    Returns:
        True if valid, False otherwise
    """
    is_valid, _ = FilenameGenerator.validate(filepath)
    return is_valid
