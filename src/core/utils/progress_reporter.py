"""Progress reporting utilities with standardized formatting."""

from typing import Optional, Any, Iterator
from tqdm import tqdm


class ProgressReporter:
    """Standardized progress reporting with tqdm."""

    def __init__(
        self,
        total: int,
        desc: str = "Processing",
        unit: str = "item",
        show_progress: bool = True,
        ncols: int = 120,
    ):
        """
        Initialize progress reporter.

        Args:
            total: Total number of items
            desc: Description text
            unit: Unit name (e.g., "item", "image", "batch")
            show_progress: Whether to show progress bar
            ncols: Width of progress bar
        """
        self.total = total
        self.desc = desc
        self.unit = unit
        self.show_progress = show_progress
        self.ncols = ncols
        self._pbar: Optional[tqdm] = None

    def __enter__(self) -> 'ProgressReporter':
        """Enter context manager."""
        if self.show_progress:
            self._pbar = tqdm(
                total=self.total,
                desc=self.desc,
                unit=self.unit,
                ncols=self.ncols,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}'
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        if self._pbar:
            self._pbar.close()

    def update(self, n: int = 1):
        """
        Update progress by n items.

        Args:
            n: Number of items to increment
        """
        if self._pbar:
            self._pbar.update(n)

    def set_postfix(self, **kwargs):
        """
        Set postfix status information.

        Args:
            **kwargs: Key-value pairs to display
        """
        if self._pbar:
            self._pbar.set_postfix(**kwargs)

    def set_postfix_str(self, s: str):
        """
        Set postfix as string.

        Args:
            s: Postfix string
        """
        if self._pbar:
            self._pbar.set_postfix_str(s)

    @staticmethod
    def iterate(
        iterable,
        desc: str = "Processing",
        unit: str = "item",
        show_progress: bool = True,
        total: Optional[int] = None,
    ) -> Iterator[Any]:
        """
        Create progress iterator (convenience method).

        Args:
            iterable: Iterable to wrap
            desc: Description text
            unit: Unit name
            show_progress: Whether to show progress
            total: Total items (if not inferrable from iterable)

        Returns:
            Iterator with progress reporting

        Example:
            for item in ProgressReporter.iterate(items, desc="Processing"):
                process(item)
        """
        if show_progress:
            return tqdm(
                iterable,
                desc=desc,
                unit=unit,
                total=total,
                ncols=120,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}'
            )
        return iterable


def format_gpu_stats(memory_gb: float, utilization_percent: Optional[float] = None) -> str:
    """
    Format GPU statistics for progress display.

    Args:
        memory_gb: Memory usage in GB
        utilization_percent: GPU utilization percentage

    Returns:
        Formatted string
    """
    parts = [f"GPU: {memory_gb:.1f}GB"]
    if utilization_percent is not None:
        parts.append(f"{utilization_percent:.0f}%")
    return " ".join(parts)


def format_analysis_stats(
    scene: Optional[str] = None,
    objects: Optional[int] = None,
    memory_gb: Optional[float] = None
) -> str:
    """
    Format analysis statistics for progress display.

    Args:
        scene: Primary scene classification
        objects: Number of objects detected
        memory_gb: GPU memory usage

    Returns:
        Formatted string
    """
    parts = []

    if scene:
        parts.append(f"Scene: {scene}")

    if objects is not None:
        parts.append(f"Objects: {objects}")

    if memory_gb is not None:
        parts.append(f"GPU: {memory_gb:.1f}GB")

    return " | ".join(parts) if parts else ""
