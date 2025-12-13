"""Face detection and analysis module using InsightFace.

Provides face detection, counting, and face encoding for recognition.
Uses InsightFace (ONNX-based) for robust, GPU-accelerated face detection.
"""

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import numpy as np

from ..interfaces.analyzers import ContentAnalyzer as AnalyzerInterface
from ..utils.logging_config import get_logger

logger = get_logger(__name__)

# Try to import insightface
try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    logger.debug("insightface not installed - face detection disabled")


class FaceDetector:
    """
    Face detection and analysis using InsightFace (ONNX runtime).

    Features:
    - Face detection with bounding boxes
    - Face count per image
    - Face encoding for recognition/clustering (128D embeddings)
    - Landmark detection (eyes, nose, mouth)
    - GPU acceleration via ONNX runtime
    """

    def __init__(
        self,
        model: str = "buffalo_sc",
        compute_encodings: bool = True,
        use_gpu: bool = True,
    ):
        """
        Initialize face detector.

        Args:
            model: Model name ("buffalo_l" for high accuracy, "buffalo_sc" for fast/small)
            compute_encodings: Whether to compute face encodings (512D vector per face)
            use_gpu: Whether to use GPU acceleration (requires CUDA)
        """
        if not INSIGHTFACE_AVAILABLE:
            logger.warning("insightface not installed. Install with: pip install insightface onnxruntime-gpu")
            self.app = None
            return

        self.model = model
        self.compute_encodings = compute_encodings
        self.use_gpu = use_gpu

        # Initialize InsightFace app
        try:
            self.app = FaceAnalysis(
                name=model,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
            )
            self.app.prepare(ctx_id=0 if use_gpu else -1, det_size=(640, 640))
            logger.debug(f"InsightFace initialized: model={model}, GPU={use_gpu}")
        except Exception as e:
            logger.error(f"Failed to initialize InsightFace: {e}")
            self.app = None

    def detect_faces_from_array(self, image_array: np.ndarray) -> Dict[str, Any]:
        """
        Detect faces in an image from numpy array.

        Args:
            image_array: RGB image as numpy array (H, W, 3)

        Returns:
            Dictionary with face detection results:
            - face_count: Number of faces detected
            - face_locations: List of bounding box dicts
            - face_landmarks: List of landmark arrays (if available)
            - face_encodings: List of 512D face embeddings (if enabled)
            - has_faces: Boolean indicating if any faces found
        """
        if not INSIGHTFACE_AVAILABLE or self.app is None:
            return {
                "face_count": 0,
                "face_locations": [],
                "has_faces": False,
                "error": "insightface not available",
            }

        try:
            # InsightFace expects BGR format
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                # Convert RGB to BGR
                bgr_array = image_array[:, :, ::-1]
            else:
                bgr_array = image_array

            # Detect faces
            faces = self.app.get(bgr_array)
            face_count = len(faces)

            result = {
                "face_count": face_count,
                "has_faces": face_count > 0,
            }

            # Extract face locations (bounding boxes)
            face_locations = []
            for face in faces:
                bbox = face.bbox.astype(int)
                # Convert from (x1, y1, x2, y2) to {left, top, right, bottom}
                face_locations.append({
                    "left": int(bbox[0]),
                    "top": int(bbox[1]),
                    "right": int(bbox[2]),
                    "bottom": int(bbox[3]),
                })
            result["face_locations"] = face_locations

            # Extract face landmarks (5 keypoints: left eye, right eye, nose, mouth_left, mouth_right)
            if face_count > 0:
                try:
                    landmarks = []
                    for face in faces:
                        if hasattr(face, 'kps') and face.kps is not None:
                            # kps is (5, 2) array of keypoints
                            landmarks.append(face.kps.tolist())
                    result["face_landmarks"] = landmarks
                except Exception as e:
                    logger.debug(f"Could not extract landmarks: {e}")

            # Extract face encodings if requested
            if self.compute_encodings and face_count > 0:
                try:
                    encodings = []
                    for face in faces:
                        if hasattr(face, 'embedding') and face.embedding is not None:
                            # InsightFace provides 512D embeddings
                            encodings.append(face.embedding.tolist())
                    result["face_encodings"] = encodings
                except Exception as e:
                    logger.debug(f"Could not extract encodings: {e}")

            return result

        except Exception as e:
            logger.warning(f"Error detecting faces: {e}")
            return {
                "face_count": 0,
                "face_locations": [],
                "has_faces": False,
                "error": str(e),
            }

    def detect_faces(self, image_path: str) -> Dict[str, Any]:
        """
        Detect faces in an image file.

        Note: For HEIC support, use detect_faces_from_array() with image_io.load_image()

        Args:
            image_path: Path to the image file

        Returns:
            Dictionary with face detection results
        """
        if not INSIGHTFACE_AVAILABLE or self.app is None:
            return {
                "face_count": 0,
                "face_locations": [],
                "has_faces": False,
                "error": "insightface not available",
            }

        try:
            # Load image using OpenCV (BGR format)
            import cv2
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Failed to load image: {image_path}")

            # Convert BGR to RGB for consistency
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return self.detect_faces_from_array(img_rgb)

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

        return results

    def compare_faces(
        self,
        encoding1: List[float],
        encoding2: List[float],
        tolerance: float = 0.6
    ) -> Tuple[bool, float]:
        """
        Compare two face encodings.

        Args:
            encoding1: First face encoding (512D vector)
            encoding2: Second face encoding (512D vector)
            tolerance: Similarity threshold (lower = more strict)

        Returns:
            Tuple of (is_match, distance)
        """
        if not INSIGHTFACE_AVAILABLE:
            return False, 1.0

        try:
            import numpy as np
            enc1 = np.array(encoding1)
            enc2 = np.array(encoding2)

            # Compute cosine similarity
            distance = 1 - np.dot(enc1, enc2) / (np.linalg.norm(enc1) * np.linalg.norm(enc2))
            is_match = distance <= tolerance

            return is_match, float(distance)

        except Exception as e:
            logger.error(f"Error comparing faces: {e}")
            return False, 1.0

    def cluster_faces(
        self,
        encodings: List[List[float]],
        tolerance: float = 0.6
    ) -> List[int]:
        """
        Cluster face encodings by similarity.

        Args:
            encodings: List of face encodings
            tolerance: Clustering threshold

        Returns:
            List of cluster labels (same label = same person)
        """
        if not INSIGHTFACE_AVAILABLE or len(encodings) == 0:
            return []

        try:
            from sklearn.cluster import DBSCAN
            import numpy as np

            # Convert to numpy array
            encodings_array = np.array(encodings)

            # Use DBSCAN for clustering
            clusterer = DBSCAN(metric="cosine", eps=tolerance, min_samples=1)
            labels = clusterer.fit_predict(encodings_array)

            return labels.tolist()

        except Exception as e:
            logger.error(f"Error clustering faces: {e}")
            return list(range(len(encodings)))  # Each face in own cluster


class FaceAnalyzer(AnalyzerInterface):
    """
    Face analyzer for integration with analysis pipeline.
    """

    def __init__(
        self,
        model: str = "buffalo_sc",
        compute_encodings: bool = False,
        use_gpu: bool = True
    ):
        """
        Initialize face analyzer.

        Args:
            model: InsightFace model name
            compute_encodings: Whether to compute face encodings
            use_gpu: Whether to use GPU acceleration
        """
        self.detector = FaceDetector(
            model=model,
            compute_encodings=compute_encodings,
            use_gpu=use_gpu
        )

    def analyze(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze image for faces.

        Args:
            image_path: Path to image file

        Returns:
            Dictionary with "faces" key containing detection results
        """
        result = self.detector.detect_faces(image_path)
        return {"faces": result}


def is_available() -> bool:
    """Check if InsightFace is available."""
    return INSIGHTFACE_AVAILABLE


# Alias for backward compatibility
is_face_detection_available = is_available
