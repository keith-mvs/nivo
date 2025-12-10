"""Hashing utilities for file and image comparison."""

import hashlib
from pathlib import Path
from typing import Literal

HashAlgorithm = Literal["md5", "sha256", "sha1"]


def file_hash(file_path: str, algorithm: HashAlgorithm = "sha256") -> str:
    """
    Calculate hash of file contents.

    Args:
        file_path: Path to file
        algorithm: Hash algorithm to use

    Returns:
        Hexadecimal hash string
    """
    hash_func = getattr(hashlib, algorithm)()

    with open(file_path, 'rb') as f:
        # Read in chunks for memory efficiency
        for chunk in iter(lambda: f.read(8192), b''):
            hash_func.update(chunk)

    return hash_func.hexdigest()


def quick_hash(file_path: str) -> str:
    """
    Calculate quick hash using file size and partial content.
    Useful for fast initial duplicate detection.

    Args:
        file_path: Path to file

    Returns:
        Hash string
    """
    path = Path(file_path)
    size = path.stat().st_size

    # Read first and last 8KB
    sample_size = min(8192, size)

    hash_func = hashlib.md5()
    hash_func.update(str(size).encode())

    with open(file_path, 'rb') as f:
        # First chunk
        hash_func.update(f.read(sample_size))

        # Last chunk if file is large enough
        if size > sample_size * 2:
            f.seek(-sample_size, 2)
            hash_func.update(f.read(sample_size))

    return hash_func.hexdigest()
