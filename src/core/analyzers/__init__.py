"""Image analysis modules."""

# Register HEIF/HEIC opener for PIL globally
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass  # pillow-heif not installed

from .face_detection import FaceAnalyzer, FaceDetector, is_available as is_face_detection_available

__all__ = [
    "FaceAnalyzer",
    "FaceDetector",
    "is_face_detection_available",
]
