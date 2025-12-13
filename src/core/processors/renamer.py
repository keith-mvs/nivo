"""Intelligent file renaming with date-based patterns."""

import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Set
from collections import defaultdict
from ..utils.logging_config import get_logger



logger = get_logger(__name__)
class ImageRenamer:
    """Rename images using intelligent patterns."""

    def __init__(
        self,
        pattern: str = "{datetime}",
        date_format: str = "%Y-%m-%d",
        time_format: str = "%H%M%S",
        datetime_format: str = "%Y-%m-%d_%H%M%S",
        collision_suffix: str = "_{seq:03d}",
        preserve_original: bool = True,
        max_filename_length: int = 200,
    ):
        """
        Initialize renamer.

        Args:
            pattern: Naming pattern with variables ({date}, {time}, {datetime}, {camera}, {seq}, {tags})
            date_format: strftime format for {date}
            time_format: strftime format for {time}
            datetime_format: strftime format for {datetime}
            collision_suffix: Suffix for name collisions
            preserve_original: Create backup before renaming
            max_filename_length: Maximum filename length
        """
        self.pattern = pattern
        self.date_format = date_format
        self.time_format = time_format
        self.datetime_format = datetime_format
        self.collision_suffix = collision_suffix
        self.preserve_original = preserve_original
        self.max_filename_length = max_filename_length

    def rename_files(
        self,
        file_metadata: List[Dict],
        output_dir: Optional[str] = None,
        dry_run: bool = True,
    ) -> Dict[str, str]:
        """
        Rename files based on metadata.

        Args:
            file_metadata: List of metadata dicts from analyzers
            output_dir: Optional output directory (uses source dir if None)
            dry_run: Preview changes without renaming

        Returns:
            Dictionary mapping old_path -> new_path
        """
        rename_map = {}
        used_names: Set[str] = set()

        for metadata in file_metadata:
            try:
                old_path = metadata.get("file_path") or metadata.get("image_path")
                if not old_path or not os.path.exists(old_path):
                    continue

                # Generate new name
                new_name = self._generate_name(metadata, used_names)

                # Determine output directory
                if output_dir:
                    new_path = str(Path(output_dir) / new_name)
                else:
                    new_path = str(Path(old_path).parent / new_name)

                # Handle collisions
                new_path = self._handle_collision(new_path, used_names)

                used_names.add(Path(new_path).name)
                rename_map[old_path] = new_path

            except Exception as e:
                logger.error(f"Error generating name for {old_path}: {e}")

        # Execute renames
        if not dry_run:
            self._execute_renames(rename_map, output_dir)
        else:
            self._preview_renames(rename_map)

        return rename_map

    def _generate_name(self, metadata: Dict, used_names: Set[str]) -> str:
        """Generate new filename from pattern and metadata."""
        # Extract datetime
        dt = self._extract_datetime(metadata)

        # Build replacement dict
        replacements = {
            "date": dt.strftime(self.date_format),
            "time": dt.strftime(self.time_format),
            "datetime": dt.strftime(self.datetime_format),
            "year": dt.strftime("%Y"),
            "month": dt.strftime("%m"),
            "day": dt.strftime("%d"),
        }

        # Camera info
        if "model" in metadata:
            camera = metadata["model"].replace(" ", "-")
            replacements["camera"] = camera

        # Tags
        if "tags" in metadata and isinstance(metadata["tags"], list):
            tags = "_".join(metadata["tags"][:3])  # First 3 tags
            replacements["tags"] = self._sanitize_filename(tags)
        elif "tag_string" in metadata:
            tags = metadata["tag_string"].replace(", ", "_")
            replacements["tags"] = self._sanitize_filename(tags)

        # Generate base name
        name = self.pattern
        for key, value in replacements.items():
            name = name.replace(f"{{{key}}}", str(value))

        # Get original extension
        original_path = metadata.get("file_path") or metadata.get("image_path")
        ext = Path(original_path).suffix.lower()

        # Combine and sanitize
        full_name = f"{name}{ext}"
        full_name = self._sanitize_filename(full_name)

        # Enforce length limit
        if len(full_name) > self.max_filename_length:
            name_part = full_name[:-len(ext)]
            name_part = name_part[:self.max_filename_length - len(ext) - 10]
            full_name = f"{name_part}{ext}"

        return full_name

    def _extract_datetime(self, metadata: Dict) -> datetime:
        """Extract datetime from metadata with fallbacks."""
        # Try EXIF datetime original
        if "datetime_original" in metadata:
            dt = metadata["datetime_original"]
            if isinstance(dt, datetime):
                return dt
            elif isinstance(dt, str):
                try:
                    return datetime.strptime(dt, "%Y:%m:%d %H:%M:%S")
                except ValueError:
                    pass

        # Try datetime modified
        if "datetime_modified" in metadata:
            dt = metadata["datetime_modified"]
            if isinstance(dt, datetime):
                return dt

        # Fallback to current time
        return datetime.now()

    def _sanitize_filename(self, name: str) -> str:
        """Remove invalid characters from filename."""
        # Remove invalid Windows/Unix characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            name = name.replace(char, '')

        # Replace spaces with underscores
        name = name.replace(' ', '_')

        # Remove consecutive underscores
        while '__' in name:
            name = name.replace('__', '_')

        return name.strip('_')

    def _handle_collision(self, path: str, used_names: Set[str]) -> str:
        """Handle filename collisions by adding sequence number."""
        if Path(path).name not in used_names:
            return path

        base_path = Path(path)
        stem = base_path.stem
        ext = base_path.suffix
        parent = base_path.parent

        seq = 1
        while True:
            suffix = self.collision_suffix.format(seq=seq)
            new_name = f"{stem}{suffix}{ext}"
            new_path = str(parent / new_name)

            if Path(new_path).name not in used_names:
                return new_path

            seq += 1
            if seq > 9999:  # Safety limit
                raise ValueError(f"Too many collisions for {path}")

    def _execute_renames(self, rename_map: Dict[str, str], output_dir: Optional[str]):
        """Execute file renames."""
        # Create output directory if needed
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        success_count = 0
        error_count = 0

        for old_path, new_path in rename_map.items():
            try:
                # Create backup if preserving
                if self.preserve_original and not output_dir:
                    backup_path = f"{old_path}.backup"
                    shutil.copy2(old_path, backup_path)

                # Rename/move file
                if output_dir:
                    shutil.copy2(old_path, new_path)  # Copy to new location
                else:
                    os.rename(old_path, new_path)  # Rename in place

                success_count += 1

            except Exception as e:
                logger.error(f"Error renaming {old_path} -> {new_path}: {e}")
                error_count += 1

        logger.info(f"Renamed {success_count} files successfully")
        if error_count > 0:
            logger.error(f"Failed to rename {error_count} files")

    def _preview_renames(self, rename_map: Dict[str, str]):
        """Preview renames without executing."""
        logger.info("=== Rename Preview ===")
        logger.info(f"Total files to rename: {len(rename_map)}\n")

        for i, (old_path, new_path) in enumerate(rename_map.items(), 1):
            old_name = Path(old_path).name
            new_name = Path(new_path).name

            if i <= 20:  # Show first 20
                logger.info(f"{old_name}")
                logger.info(f"  -> {new_name}\n")

        if len(rename_map) > 20:
            logger.info(f"... and {len(rename_map) - 20} more files")

        logger.info("Run without --dry-run to execute renames")
