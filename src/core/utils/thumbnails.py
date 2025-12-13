"""Fast thumbnail generation with caching."""

from pathlib import Path
from typing import Optional, Tuple
import hashlib
from PIL import Image
from .image_io import load_image
from .logging_config import get_logger

logger = get_logger(__name__)


class ThumbnailGenerator:
    """Generate and cache image thumbnails."""

    def __init__(
        self,
        size: Tuple[int, int] = (256, 256),
        cache_dir: Optional[Path] = None,
        format: str = "webp",
        quality: int = 85,
    ):
        """
        Initialize thumbnail generator.

        Args:
            size: Thumbnail dimensions (width, height)
            cache_dir: Cache directory (default: .thumbnails/)
            format: Output format (webp, jpeg, png)
            quality: Compression quality (1-100)
        """
        self.size = size
        self.cache_dir = cache_dir or Path(".thumbnails")
        self.format = format.lower()
        self.quality = quality
        self.cache_dir.mkdir(exist_ok=True)

    def _get_cache_path(self, image_path: str) -> Path:
        """Get cached thumbnail path for image."""
        # Use MD5 of absolute path for cache key
        path_hash = hashlib.md5(str(Path(image_path).absolute()).encode()).hexdigest()
        return self.cache_dir / f"{path_hash[:16]}.{self.format}"

    def generate(
        self,
        image_path: str,
        force: bool = False,
    ) -> Optional[Path]:
        """
        Generate thumbnail for image.

        Args:
            image_path: Path to source image
            force: Regenerate even if cached

        Returns:
            Path to thumbnail or None if failed
        """
        try:
            cache_path = self._get_cache_path(image_path)

            # Return cached if exists and not forcing regeneration
            if cache_path.exists() and not force:
                return cache_path

            # Load image (supports HEIC/HEIF)
            img = load_image(image_path)
            if img is None:
                logger.warning(f"Failed to load: {image_path}")
                return None

            # Convert to RGB if needed
            if img.mode in ("RGBA", "LA", "P"):
                img = img.convert("RGB")

            # Resize preserving aspect ratio
            img.thumbnail(self.size, Image.Resampling.LANCZOS)

            # Save thumbnail
            save_kwargs = {"quality": self.quality} if self.format in ("jpeg", "webp") else {}
            img.save(cache_path, format=self.format.upper(), **save_kwargs)

            return cache_path

        except Exception as e:
            logger.error(f"Error generating thumbnail for {image_path}: {e}")
            return None

    def generate_batch(
        self,
        image_paths: list,
        show_progress: bool = True,
    ) -> dict:
        """
        Generate thumbnails for multiple images.

        Args:
            image_paths: List of image paths
            show_progress: Show progress indicator

        Returns:
            Dict mapping image_path -> thumbnail_path
        """
        thumbnails = {}

        for i, image_path in enumerate(image_paths):
            if show_progress and (i + 1) % 50 == 0:
                logger.info(f"Generated {i + 1}/{len(image_paths)} thumbnails")

            thumb_path = self.generate(image_path)
            if thumb_path:
                thumbnails[image_path] = thumb_path

        logger.info(f"Generated {len(thumbnails)}/{len(image_paths)} thumbnails")
        return thumbnails

    def clear_cache(self) -> int:
        """Clear all cached thumbnails. Returns count deleted."""
        count = 0
        for thumb in self.cache_dir.glob(f"*.{self.format}"):
            thumb.unlink()
            count += 1
        return count


def create_thumbnail(
    image_path: str,
    size: Tuple[int, int] = (256, 256),
) -> Optional[Path]:
    """Quick helper: Generate single thumbnail."""
    generator = ThumbnailGenerator(size=size)
    return generator.generate(image_path)
