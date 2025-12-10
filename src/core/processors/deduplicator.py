"""Deduplication system using file hashing.

Efficiently finds and handles duplicate photos using multi-threaded hashing.
"""

import os
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import concurrent.futures
from tqdm import tqdm

from ..utils.hash_utils import file_hash, quick_hash
from ..utils.image_io import get_image_dimensions


class Deduplicator:
    """Find and handle duplicate images."""

    def __init__(
        self,
        hash_algorithm: str = "sha256",
        use_quick_hash: bool = True,
        max_workers: int = 4,
    ):
        """
        Initialize deduplicator.

        Args:
            hash_algorithm: Hash algorithm (md5, sha256, sha1)
            use_quick_hash: Use quick hash for initial screening
            max_workers: Number of parallel workers
        """
        self.hash_algorithm = hash_algorithm
        self.use_quick_hash = use_quick_hash
        self.max_workers = max_workers

    def find_duplicates(
        self,
        file_paths: List[str],
        show_progress: bool = True,
    ) -> Dict[str, List[str]]:
        """
        Find duplicate files.

        Args:
            file_paths: List of file paths to check
            show_progress: Show progress bar

        Returns:
            Dictionary mapping hash -> list of duplicate file paths
        """
        print(f"Scanning {len(file_paths)} files for duplicates...")

        # Step 1: Quick hash screening (if enabled)
        if self.use_quick_hash and len(file_paths) > 100:
            file_paths = self._quick_hash_screening(file_paths, show_progress)
            print(f"Quick screening reduced candidates to {len(file_paths)} files")

        # Step 2: Full hash computation
        hash_map = self._compute_hashes(file_paths, show_progress)

        # Step 3: Find duplicates
        duplicates = {
            hash_val: paths
            for hash_val, paths in hash_map.items()
            if len(paths) > 1
        }

        if duplicates:
            total_dupes = sum(len(paths) - 1 for paths in duplicates.values())
            print(f"Found {len(duplicates)} sets of duplicates ({total_dupes} duplicate files)")
        else:
            print("No duplicates found")

        return duplicates

    def _quick_hash_screening(
        self,
        file_paths: List[str],
        show_progress: bool,
    ) -> List[str]:
        """
        Use quick hash to eliminate obvious non-duplicates.

        Returns files that might be duplicates (same quick hash).
        """
        quick_hash_map = defaultdict(list)

        iterator = tqdm(file_paths, desc="Quick scan") if show_progress else file_paths

        # Compute quick hashes in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_path = {
                executor.submit(quick_hash, path): path
                for path in file_paths
            }

            for future in concurrent.futures.as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    qhash = future.result()
                    quick_hash_map[qhash].append(path)
                except Exception as e:
                    print(f"Error quick hashing {path}: {e}")

        # Return only files with potential duplicates
        potential_dupes = []
        for paths in quick_hash_map.values():
            if len(paths) > 1:
                potential_dupes.extend(paths)

        return potential_dupes

    def _compute_hashes(
        self,
        file_paths: List[str],
        show_progress: bool,
    ) -> Dict[str, List[str]]:
        """Compute full hashes in parallel."""
        hash_map = defaultdict(list)

        iterator = tqdm(file_paths, desc="Computing hashes") if show_progress else file_paths

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_path = {
                executor.submit(file_hash, path, self.hash_algorithm): path
                for path in file_paths
            }

            for future in concurrent.futures.as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    hash_val = future.result()
                    hash_map[hash_val].append(path)
                except Exception as e:
                    print(f"Error hashing {path}: {e}")

        return hash_map

    def select_best_to_keep(
        self,
        duplicate_paths: List[str],
        strategy: str = "highest_quality",
    ) -> str:
        """
        Select which duplicate to keep.

        Args:
            duplicate_paths: List of duplicate file paths
            strategy: Selection strategy (highest_quality, oldest, newest, largest)

        Returns:
            Path of file to keep
        """
        if strategy == "highest_quality":
            return self._keep_highest_quality(duplicate_paths)
        elif strategy == "oldest":
            return min(duplicate_paths, key=lambda p: Path(p).stat().st_mtime)
        elif strategy == "newest":
            return max(duplicate_paths, key=lambda p: Path(p).stat().st_mtime)
        elif strategy == "largest":
            return max(duplicate_paths, key=lambda p: Path(p).stat().st_size)
        else:
            return duplicate_paths[0]

    def _keep_highest_quality(self, paths: List[str]) -> str:
        """Select highest quality image based on resolution and file size."""
        best_path = paths[0]
        best_score = 0

        for path in paths:
            score = 0

            # Resolution score
            dims = get_image_dimensions(path)
            if dims:
                width, height = dims
                score += width * height

            # File size score (larger usually means less compression)
            file_size = Path(path).stat().st_size
            score += file_size / 1000  # Convert to KB and add

            if score > best_score:
                best_score = score
                best_path = path

        return best_path

    def remove_duplicates(
        self,
        duplicates: Dict[str, List[str]],
        strategy: str = "highest_quality",
        dry_run: bool = True,
    ) -> Dict[str, any]:
        """
        Remove duplicate files.

        Args:
            duplicates: Dictionary from find_duplicates()
            strategy: Which file to keep
            dry_run: If True, don't actually delete files

        Returns:
            Dictionary with deletion statistics
        """
        stats = {
            "files_to_delete": [],
            "files_to_keep": [],
            "space_saved": 0,
            "dry_run": dry_run,
        }

        for hash_val, paths in duplicates.items():
            # Select file to keep
            keep_path = self.select_best_to_keep(paths, strategy)
            stats["files_to_keep"].append(keep_path)

            # Mark others for deletion
            for path in paths:
                if path != keep_path:
                    file_size = Path(path).stat().st_size
                    stats["files_to_delete"].append(path)
                    stats["space_saved"] += file_size

                    # Delete if not dry run
                    if not dry_run:
                        try:
                            os.remove(path)
                            print(f"Deleted: {path}")
                        except Exception as e:
                            print(f"Error deleting {path}: {e}")

        if dry_run:
            print(f"\nDry run: Would delete {len(stats['files_to_delete'])} files")
            print(f"Space that would be saved: {stats['space_saved'] / 1_000_000:.2f} MB")
        else:
            print(f"\nDeleted {len(stats['files_to_delete'])} duplicate files")
            print(f"Space saved: {stats['space_saved'] / 1_000_000:.2f} MB")

        return stats

    def generate_report(
        self,
        duplicates: Dict[str, List[str]],
        output_path: Optional[str] = None,
    ) -> str:
        """
        Generate human-readable duplicate report.

        Args:
            duplicates: Dictionary from find_duplicates()
            output_path: Optional path to save report

        Returns:
            Report text
        """
        lines = ["# Duplicate Files Report\n"]
        lines.append(f"Total duplicate sets: {len(duplicates)}\n")

        total_dupes = sum(len(paths) - 1 for paths in duplicates.values())
        total_waste = sum(
            sum(Path(p).stat().st_size for p in paths[1:])
            for paths in duplicates.values()
        )

        lines.append(f"Total duplicate files: {total_dupes}")
        lines.append(f"Wasted space: {total_waste / 1_000_000:.2f} MB\n")

        for i, (hash_val, paths) in enumerate(duplicates.items(), 1):
            lines.append(f"\n## Duplicate Set {i}")
            lines.append(f"Hash: {hash_val}")
            lines.append(f"Files ({len(paths)}):")

            for path in sorted(paths):
                size = Path(path).stat().st_size / 1000
                lines.append(f"  - {path} ({size:.1f} KB)")

        report = "\n".join(lines)

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"Report saved to: {output_path}")

        return report
