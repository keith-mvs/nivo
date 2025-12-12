"""Memory-aware LRU cache for loaded images with Windows Unicode support.

Reduces redundant image loading across pipeline phases by caching PIL Image objects.
Implements memory-aware eviction to prevent OOM errors.
"""

import os
import psutil
from pathlib import Path
from typing import Optional, Dict, Tuple
from collections import OrderedDict
from PIL import Image
import numpy as np


class ImageCache:
    """LRU cache for loaded images with memory management."""

    def __init__(
        self,
        max_memory_mb: int = 500,
        max_items: int = 100,
        enable_cache: bool = True
    ):
        """
        Initialize image cache.

        Args:
            max_memory_mb: Maximum cache memory in MB
            max_items: Maximum number of cached images
            enable_cache: Enable/disable caching (for testing)
        """
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.max_items = max_items
        self.enable_cache = enable_cache

        # LRU cache: {path: (image, size_bytes)}
        self._cache: OrderedDict[str, Tuple[Image.Image, int]] = OrderedDict()
        self._current_memory = 0

        # Statistics
        self._hits = 0
        self._misses = 0

    def get(self, image_path: str) -> Optional[Image.Image]:
        """
        Get image from cache or load if not cached.

        Args:
            image_path: Path to image file

        Returns:
            PIL Image or None on error
        """
        if not self.enable_cache:
            return self._load_image(image_path)

        # Normalize path for cache key
        cache_key = str(Path(image_path).resolve())

        # Check cache
        if cache_key in self._cache:
            self._hits += 1
            # Move to end (most recently used)
            self._cache.move_to_end(cache_key)
            img, size = self._cache[cache_key]
            return img.copy()  # Return copy to prevent mutations

        # Cache miss - load image
        self._misses += 1
        img = self._load_image(image_path)

        if img is not None:
            self._add_to_cache(cache_key, img)

        return img

    def _load_image(self, image_path: str) -> Optional[Image.Image]:
        """
        Load image with Windows Unicode filename support.

        Args:
            image_path: Path to image file

        Returns:
            PIL Image or None on error
        """
        try:
            # PIL.Image.open handles Unicode paths correctly on Windows
            img = Image.open(image_path)
            # Convert to RGB for consistency
            if img.mode in ('RGBA', 'LA', 'P'):
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode in ('RGBA', 'LA'):
                    background.paste(img, mask=img.split()[-1])
                    img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')

            return img

        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

    def _add_to_cache(self, cache_key: str, img: Image.Image):
        """
        Add image to cache with memory management.

        Args:
            cache_key: Normalized cache key
            img: PIL Image to cache
        """
        # Calculate image memory size
        img_size = self._estimate_image_size(img)

        # Evict if necessary
        self._evict_if_needed(img_size)

        # Add to cache
        self._cache[cache_key] = (img.copy(), img_size)
        self._current_memory += img_size

    def _estimate_image_size(self, img: Image.Image) -> int:
        """
        Estimate memory size of PIL Image in bytes.

        Args:
            img: PIL Image

        Returns:
            Estimated size in bytes
        """
        # Estimate: width * height * bytes_per_pixel
        width, height = img.size
        mode_to_bytes = {
            'RGB': 3,
            'RGBA': 4,
            'L': 1,
            'LA': 2,
            'P': 1,
        }
        bytes_per_pixel = mode_to_bytes.get(img.mode, 4)
        return width * height * bytes_per_pixel

    def _evict_if_needed(self, required_space: int):
        """
        Evict least recently used images if needed.

        Args:
            required_space: Space needed for new image in bytes
        """
        # Evict based on item count
        while len(self._cache) >= self.max_items:
            self._evict_oldest()

        # Evict based on memory
        while (self._current_memory + required_space > self.max_memory_bytes
               and len(self._cache) > 0):
            self._evict_oldest()

    def _evict_oldest(self):
        """Evict least recently used image."""
        if not self._cache:
            return

        # Pop oldest (first item in OrderedDict)
        key, (img, size) = self._cache.popitem(last=False)
        self._current_memory -= size

    def clear(self):
        """Clear all cached images."""
        self._cache.clear()
        self._current_memory = 0

    def get_stats(self) -> Dict[str, any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        total_accesses = self._hits + self._misses
        hit_rate = (self._hits / total_accesses * 100) if total_accesses > 0 else 0

        return {
            "enabled": self.enable_cache,
            "items_cached": len(self._cache),
            "memory_used_mb": self._current_memory / (1024 * 1024),
            "memory_limit_mb": self.max_memory_bytes / (1024 * 1024),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
        }

    def print_stats(self):
        """Print cache statistics."""
        stats = self.get_stats()
        print(f"\n=== Image Cache Statistics ===")
        print(f"Enabled: {stats['enabled']}")
        print(f"Items cached: {stats['items_cached']}/{self.max_items}")
        print(f"Memory used: {stats['memory_used_mb']:.1f}/{stats['memory_limit_mb']:.1f} MB")
        print(f"Hit rate: {stats['hit_rate']:.1f}% ({stats['hits']} hits / {stats['misses']} misses)")

    def __contains__(self, image_path: str) -> bool:
        """Check if image is in cache."""
        cache_key = str(Path(image_path).resolve())
        return cache_key in self._cache

    def __len__(self) -> int:
        """Get number of cached images."""
        return len(self._cache)


# Global cache instance (lazy initialization)
_global_cache: Optional[ImageCache] = None


def get_cache(
    max_memory_mb: int = 500,
    max_items: int = 100,
    enable_cache: bool = True
) -> ImageCache:
    """
    Get or create global image cache instance.

    Args:
        max_memory_mb: Maximum cache memory in MB
        max_items: Maximum number of cached images
        enable_cache: Enable/disable caching

    Returns:
        Global ImageCache instance
    """
    global _global_cache

    if _global_cache is None:
        _global_cache = ImageCache(
            max_memory_mb=max_memory_mb,
            max_items=max_items,
            enable_cache=enable_cache
        )

    return _global_cache


def clear_cache():
    """Clear global image cache."""
    global _global_cache
    if _global_cache is not None:
        _global_cache.clear()


def get_cached_image(image_path: str) -> Optional[Image.Image]:
    """
    Convenience function to get image from global cache.

    Args:
        image_path: Path to image file

    Returns:
        PIL Image or None
    """
    cache = get_cache()
    return cache.get(image_path)
