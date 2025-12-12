"""Abstract base classes for image analyzers."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List


class ImageAnalyzer(ABC):
    """Base interface for all image analyzers."""

    @abstractmethod
    def analyze(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze a single image.

        Args:
            image_path: Path to image file

        Returns:
            Analysis results dictionary
        """
        pass


class MetadataAnalyzer(ImageAnalyzer):
    """Interface for metadata extraction analyzers."""

    @abstractmethod
    def extract(self, image_path: str) -> Dict[str, Any]:
        """
        Extract metadata from image.

        Args:
            image_path: Path to image file

        Returns:
            Metadata dictionary
        """
        pass

    def analyze(self, image_path: str) -> Dict[str, Any]:
        """Implement ImageAnalyzer interface."""
        return self.extract(image_path)


class ContentAnalyzer(ImageAnalyzer):
    """Interface for content-based analyzers (quality, blur, color)."""

    @abstractmethod
    def analyze(self, image_path: str) -> Dict[str, Any]:
        """
        Perform content analysis on image.

        Args:
            image_path: Path to image file

        Returns:
            Content analysis results
        """
        pass


class MLAnalyzer(ImageAnalyzer):
    """Interface for machine learning vision analyzers."""

    @abstractmethod
    def analyze(self, image_path: str) -> Dict[str, Any]:
        """
        Perform ML vision analysis on single image.

        Args:
            image_path: Path to image file

        Returns:
            ML analysis results
        """
        pass

    @abstractmethod
    def analyze_batch(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Perform batch ML vision analysis.

        Args:
            image_paths: List of image file paths

        Returns:
            List of analysis results
        """
        pass

    @abstractmethod
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current GPU memory usage.

        Returns:
            Dictionary with memory statistics
        """
        pass

    @abstractmethod
    def clear_cache(self):
        """Clear GPU cache and free memory."""
        pass
