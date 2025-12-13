"""Face detection and analysis module.

Provides face detection, counting, and optional face encoding for recognition.
Uses face_recognition library (dlib-based) for robust face detection.
"""

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from PIL import Image
import numpy as np

from ..interfaces.analyzers import ContentAnalyzer as AnalyzerInterface
from ..utils.logging_config import get_logger

logger = get_logger(__name__)

# Try to import face_recognition
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    logger.debug("face_recognition not installed - face detection disabled")


class FaceDetector:
    """
    Face detection and analysis using dlib (via face_recognition library).

    Features:
    - Face detection with bounding boxes
    - Face count per image
    - Optional face encoding for recognition/clustering
    - Landmark detection (eyes, nose, mouth)
    """

    def __init__(
        self,
        model: str = "hog",
        num_jitters: int = 1,
        compute_encodings: bool = False,
    ):
        """
        Initialize face detector.

        Args:
            model: Detection model ("hog" for CPU, "cnn" for GPU - more accurate)
            num_jitters: Number of times to re-sample face for encoding (higher = more accurate but slower)
            compute_encodings: Whether to compute face encodings (128D vector per face)
        """
        if not FACE_RECOGNITION_AVAILABLE:
            logger.warning("face_recognition not installed. Install with: pip install face-recognition")

        self.model = model
        self.num_jitters = num_jitters
        self.compute_encodings = compute_encodings

    def detect_faces(self, image_path: str) -> Dict[str, Any]:
        """
        Detect faces in an image.

        Args:
            image_path: Path to the image file

        Returns:
            Dictionary with face detection results:
            - face_count: Number of faces detected
            - face_locations: List of (top, right, bottom, left) tuples
            - face_landmarks: List of landmark dictionaries (if available)
            - face_encodings: List of 128D face encodings (if enabled)
            - has_faces: Boolean indicating if any faces found
        """
        if not FACE_RECOGNITION_AVAILABLE:
            return {
                "face_count": 0,
                "face_locations": [],
                "has_faces": False,
                "error": "face_recognition not installed",
            }

        try:
            # Load image
            image = face_recognition.load_image_file(image_path)

            # Detect face locations
            face_locations = face_recognition.face_locations(image, model=self.model)
            face_count = len(face_locations)

            result = {
                "face_count": face_count,
                "face_locations": [
                    {"top": t, "right": r, "bottom": b, "left": l}
                    for t, r, b, l in face_locations
                ],
                "has_faces": face_count > 0,
            }

            # Get face landmarks (eyes, nose, mouth, etc.)
            if face_count > 0:
                try:
                    face_landmarks_list = face_recognition.face_landmarks(image, face_locations)
                    result["face_landmarks"] = face_landmarks_list
                except Exception as e:
                    logger.debug(f"Could not extract landmarks: {e}")

            # Compute face encodings if requested
            if self.compute_encodings and face_count > 0:
                try:
                    encodings = face_recognition.face_encodings(
                        image,
                        face_locations,
                        num_jitters=self.num_jitters
                    )
                    # Convert numpy arrays to lists for JSON serialization
                    result["face_encodings"] = [enc.tolist() for enc in encodings]
                except Exception as e:
                    logger.debug(f"Could not compute encodings: {e}")

            return result

        except Exception as e:
            logger.warning(f"Error detecting faces in {image_path}: {e}")
            return {
                "face_count": 0,
                "face_locations": [],
                "has_faces": False,
                "error": str(e),
            }

    def detect_batch(
        self,
        image_paths: List[str],
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Detect faces in multiple images.

        Args:
            image_paths: List of image file paths
            show_progress: Show progress information

        Returns:
            List of face detection results
        """
        results = []

        for i, path in enumerate(image_paths):
            if show_progress and (i + 1) % 10 == 0:
                logger.info(f"Face detection: {i + 1}/{len(image_paths)}")

            result = self.detect_faces(path)
            result["file_path"] = path
            results.append(result)

        total_faces = sum(r.get("face_count", 0) for r in results)
        images_with_faces = sum(1 for r in results if r.get("has_faces", False))

        logger.info(f"Face detection complete: {total_faces} faces in {images_with_faces}/{len(image_paths)} images")

        return results

    def compare_faces(
        self,
        known_encoding: List[float],
        unknown_encoding: List[float],
        tolerance: float = 0.6
    ) -> Tuple[bool, float]:
        """
        Compare two face encodings.

        Args:
            known_encoding: Reference face encoding (128D list)
            unknown_encoding: Face encoding to compare (128D list)
            tolerance: Distance threshold for match (lower = stricter)

        Returns:
            Tuple of (is_match, distance)
        """
        if not FACE_RECOGNITION_AVAILABLE:
            return False, 1.0

        known = np.array(known_encoding)
        unknown = np.array(unknown_encoding)

        distance = np.linalg.norm(known - unknown)
        is_match = distance <= tolerance

        return is_match, float(distance)

    def cluster_faces(
        self,
        encodings: List[List[float]],
        tolerance: float = 0.6
    ) -> List[int]:
        """
        Cluster face encodings into groups (same person).

        Args:
            encodings: List of face encodings
            tolerance: Distance threshold for grouping

        Returns:
            List of cluster labels (same label = same person)
        """
        if not FACE_RECOGNITION_AVAILABLE or not encodings:
            return []

        # Convert to numpy
        encoding_arrays = [np.array(enc) for enc in encodings]

        # Simple clustering using face_recognition
        labels = face_recognition.cluster_faces(encoding_arrays, tolerance)

        return labels.tolist()


class FaceAnalyzer(AnalyzerInterface):
    """
    Analyzer interface wrapper for face detection.

    Integrates face detection into the analysis pipeline.
    """

    def __init__(
        self,
        model: str = "hog",
        compute_encodings: bool = False,
    ):
        """
        Initialize face analyzer.

        Args:
            model: Detection model ("hog" or "cnn")
            compute_encodings: Whether to compute face encodings
        """
        self.detector = FaceDetector(
            model=model,
            compute_encodings=compute_encodings,
        )
        self.enabled = FACE_RECOGNITION_AVAILABLE

    def analyze(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze faces in an image.

        Args:
            image_path: Path to the image file

        Returns:
            Dictionary with face analysis results
        """
        if not self.enabled:
            return {"faces": {"face_count": 0, "has_faces": False}}

        result = self.detector.detect_faces(image_path)
        return {"faces": result}

    def analyze_batch(
        self,
        image_paths: List[str],
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Analyze faces in multiple images.

        Args:
            image_paths: List of image file paths
            show_progress: Show progress

        Returns:
            List of face analysis results
        """
        if not self.enabled:
            return [{"faces": {"face_count": 0, "has_faces": False}} for _ in image_paths]

        results = self.detector.detect_batch(image_paths, show_progress)
        return [{"faces": r} for r in results]


def is_available() -> bool:
    """Check if face recognition is available."""
    return FACE_RECOGNITION_AVAILABLE
