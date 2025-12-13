"""SQLite-based cache for incremental image analysis.

Stores file fingerprints and analysis results to skip re-analyzing unchanged files.
"""

import sqlite3
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from contextlib import contextmanager

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class AnalysisCache:
    """SQLite cache for storing and retrieving image analysis results."""

    SCHEMA_VERSION = 1

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the analysis cache.

        Args:
            db_path: Path to SQLite database. Defaults to .nivo_cache.db in cwd
        """
        self.db_path = Path(db_path) if db_path else Path(".nivo_cache.db")
        self._init_database()

    def _init_database(self):
        """Initialize database schema if needed."""
        with self._connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_info (
                    version INTEGER PRIMARY KEY
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS file_cache (
                    file_path TEXT PRIMARY KEY,
                    file_hash TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    file_mtime REAL NOT NULL,
                    analysis_json TEXT NOT NULL,
                    cached_at TEXT NOT NULL,
                    cache_version INTEGER DEFAULT 1
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_file_hash ON file_cache(file_hash)
            """)

            # Check/update schema version
            cursor = conn.execute("SELECT version FROM schema_info LIMIT 1")
            row = cursor.fetchone()
            if row is None:
                conn.execute("INSERT INTO schema_info (version) VALUES (?)",
                           (self.SCHEMA_VERSION,))

            conn.commit()

        logger.debug(f"Analysis cache initialized: {self.db_path}")

    @contextmanager
    def _connection(self):
        """Context manager for database connection."""
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _compute_file_fingerprint(self, file_path: str) -> Dict[str, Any]:
        """
        Compute fingerprint for a file based on path, size, and mtime.

        Args:
            file_path: Path to the file

        Returns:
            Dictionary with file_hash, file_size, file_mtime
        """
        path = Path(file_path)
        stat = path.stat()

        # Quick hash based on path + size + mtime
        fingerprint = f"{path.name}:{stat.st_size}:{stat.st_mtime}"
        file_hash = hashlib.md5(fingerprint.encode()).hexdigest()

        return {
            "file_hash": file_hash,
            "file_size": stat.st_size,
            "file_mtime": stat.st_mtime,
        }

    def get_cached_result(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Get cached analysis result if file hasn't changed.

        Args:
            file_path: Path to the image file

        Returns:
            Cached analysis result or None if cache miss/stale
        """
        try:
            current_fp = self._compute_file_fingerprint(file_path)
        except FileNotFoundError:
            return None

        with self._connection() as conn:
            cursor = conn.execute("""
                SELECT analysis_json, file_hash, file_size, file_mtime
                FROM file_cache
                WHERE file_path = ?
            """, (file_path,))

            row = cursor.fetchone()

            if row is None:
                return None

            # Check if file has changed
            if (row["file_hash"] == current_fp["file_hash"] and
                row["file_size"] == current_fp["file_size"] and
                abs(row["file_mtime"] - current_fp["file_mtime"]) < 1.0):

                logger.debug(f"Cache hit: {file_path}")
                return json.loads(row["analysis_json"])

            logger.debug(f"Cache stale: {file_path}")
            return None

    def cache_result(self, file_path: str, analysis: Dict[str, Any]):
        """
        Store analysis result in cache.

        Args:
            file_path: Path to the image file
            analysis: Analysis result dictionary
        """
        try:
            fp = self._compute_file_fingerprint(file_path)
        except FileNotFoundError:
            logger.warning(f"Cannot cache result for missing file: {file_path}")
            return

        with self._connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO file_cache
                (file_path, file_hash, file_size, file_mtime, analysis_json, cached_at, cache_version)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                file_path,
                fp["file_hash"],
                fp["file_size"],
                fp["file_mtime"],
                json.dumps(analysis, default=str),
                datetime.now().isoformat(),
                self.SCHEMA_VERSION,
            ))
            conn.commit()

        logger.debug(f"Cached: {file_path}")

    def get_cached_batch(self, file_paths: List[str]) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        Get cached results for multiple files efficiently.

        Args:
            file_paths: List of file paths

        Returns:
            Dictionary mapping file_path -> cached result (or None)
        """
        results = {}

        for path in file_paths:
            results[path] = self.get_cached_result(path)

        cached = sum(1 for v in results.values() if v is not None)
        logger.info(f"Cache: {cached}/{len(file_paths)} hits ({100*cached/len(file_paths):.1f}%)")

        return results

    def cache_batch(self, results: List[Dict[str, Any]]):
        """
        Cache multiple analysis results efficiently.

        Args:
            results: List of analysis result dictionaries (must contain 'file_path')
        """
        with self._connection() as conn:
            for result in results:
                file_path = result.get("file_path")
                if not file_path:
                    continue

                try:
                    fp = self._compute_file_fingerprint(file_path)
                except FileNotFoundError:
                    continue

                conn.execute("""
                    INSERT OR REPLACE INTO file_cache
                    (file_path, file_hash, file_size, file_mtime, analysis_json, cached_at, cache_version)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    file_path,
                    fp["file_hash"],
                    fp["file_size"],
                    fp["file_mtime"],
                    json.dumps(result, default=str),
                    datetime.now().isoformat(),
                    self.SCHEMA_VERSION,
                ))

            conn.commit()

        logger.info(f"Cached {len(results)} analysis results")

    def invalidate(self, file_path: str):
        """Remove cached entry for a file."""
        with self._connection() as conn:
            conn.execute("DELETE FROM file_cache WHERE file_path = ?", (file_path,))
            conn.commit()

    def clear(self):
        """Clear all cached entries."""
        with self._connection() as conn:
            conn.execute("DELETE FROM file_cache")
            conn.commit()

        logger.info("Analysis cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._connection() as conn:
            cursor = conn.execute("""
                SELECT COUNT(*) as count,
                       SUM(file_size) as total_bytes,
                       MIN(cached_at) as oldest,
                       MAX(cached_at) as newest
                FROM file_cache
            """)
            row = cursor.fetchone()

            return {
                "entries": row["count"] or 0,
                "total_file_bytes": row["total_bytes"] or 0,
                "oldest_entry": row["oldest"],
                "newest_entry": row["newest"],
                "db_size_mb": self.db_path.stat().st_size / 1_000_000 if self.db_path.exists() else 0,
            }


# Global cache instance
_cache_instance: Optional[AnalysisCache] = None


def get_cache(db_path: Optional[str] = None) -> AnalysisCache:
    """Get global cache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = AnalysisCache(db_path)
    return _cache_instance


def reset_cache():
    """Reset global cache instance."""
    global _cache_instance
    _cache_instance = None
