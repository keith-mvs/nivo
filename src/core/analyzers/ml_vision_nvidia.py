"""NVIDIA Build API vision analysis for high-quality tagging and scene understanding.

This module provides cloud-based vision analysis using NVIDIA Build API:
- Llama 3.2 Vision (11B) for scene understanding and detailed tagging
- VLM-based object detection with natural language understanding
- Superior tagging quality vs local YOLO/CLIP models
- No GPU memory constraints (cloud API)
- Rate-limited API client with retry logic

Key Features:
- Natural language scene descriptions
- Context-aware object detection
- Searchable tag generation
- Cloud-based processing (no local GPU required)
- Automatic rate limiting and retries
"""

from PIL import Image
from typing import Dict, Any, List, Optional
from pathlib import Path
import tempfile
from tqdm import tqdm

from .base_ml_analyzer import BaseMLAnalyzer
from ...adapters.nvidia_build.vision_language import VisionLanguageModel
from ...adapters.nvidia_build.retail_detector import RetailObjectDetector
from ..utils.logging_config import get_logger


logger = get_logger(__name__)


class NVIDIAVisionAnalyzer(BaseMLAnalyzer):
    """NVIDIA Build API vision analyzer for high-quality scene and object analysis."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        use_gpu: bool = False,  # Not used, kept for API compatibility
        batch_size: int = 8,  # API rate limiting
        model: str = "llama-vision",  # "llama-vision" or "nemotron"
        min_confidence: float = 0.3,  # Lower for VLM (more permissive)
        gpu_monitor: Optional[Any] = None,  # Not used, kept for compatibility
    ):
        """
        Initialize NVIDIA Build API vision analyzer.

        Args:
            api_key: NVIDIA API key (reads from NVIDIA_API_KEY env var if None)
            use_gpu: Ignored (API is cloud-based)
            batch_size: Batch size for API requests (rate limiting)
            model: VLM model to use ("llama-vision" or "nemotron")
            min_confidence: Not used for VLM, kept for compatibility
            gpu_monitor: Ignored (API is cloud-based)
        """
        # Initialize base class (minimal setup, we don't use CLIP)
        super().__init__(
            use_gpu=False,  # API-based, no local GPU
            batch_size=batch_size,
            scene_model=None,  # We use NVIDIA VLM instead
            min_confidence=min_confidence,
            gpu_monitor=None,
        )

        # NVIDIA-specific configuration
        self.model_name = model
        self.api_key = api_key

        # Initialize NVIDIA clients
        self._vlm_model = None
        self._object_detector = None

        logger.info("NVIDIA Vision Analyzer initialized (cloud-based)")
        logger.info(f"Model: {model}")
        logger.info(f"Batch size: {batch_size}")

    def _load_detection_model(self):
        """Load NVIDIA object detection model (implements abstract method)."""
        self._load_nvidia_models()

    def _load_nvidia_models(self):
        """Initialize NVIDIA Build API clients."""
        try:
            logger.debug("Initializing NVIDIA Build API clients...")

            # Vision-language model for scene understanding
            self._vlm_model = VisionLanguageModel(
                api_key=self.api_key,
                model=self.model_name
            )

            # Object detector
            self._object_detector = RetailObjectDetector(api_key=self.api_key)

            logger.debug("NVIDIA Build API clients initialized successfully")

        except ValueError as e:
            logger.error(f"ERROR: Failed to initialize NVIDIA API: {e}")
            logger.error("Set NVIDIA_API_KEY environment variable or pass api_key parameter")
            logger.error("Get your API key at: https://build.nvidia.com")
            self._vlm_model = "FAILED"
            self._object_detector = "FAILED"
        except Exception as e:
            logger.error(f"ERROR: Failed to initialize NVIDIA models: {e}")
            self._vlm_model = "FAILED"
            self._object_detector = "FAILED"

    def _classify_scene_batch(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        """
        Classify scene type using NVIDIA VLM (overrides base class).

        Args:
            images: List of PIL images

        Returns:
            List of scene classification dictionaries
        """
        # Load models if needed
        if self._vlm_model is None:
            self._load_nvidia_models()

        if self._vlm_model == "FAILED":
            logger.warning("NVIDIA VLM not available, using fallback")
            return [{"primary_scene": "unknown", "scene_scores": {}, "scene_confidence": 0.0}] * len(images)

        results = []
        for img in images:
            try:
                # Save image temporarily for API call
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    temp_path = tmp.name
                    img.save(temp_path, "JPEG")

                try:
                    # Get scene description
                    prompt = "In 2-3 words, what type of scene is this? (e.g., 'outdoor nature', 'indoor room', 'vehicle interior', 'architecture', 'food', 'portrait')"
                    description = self._vlm_model.describe_image(
                        temp_path,
                        prompt=prompt,
                        max_tokens=10,
                        temperature=0.1
                    )

                    # Extract primary scene (first word usually most relevant)
                    scene = description.strip().split()[0].lower() if description else "unknown"
                    results.append({
                        "primary_scene": scene,
                        "scene_scores": {scene: 1.0},
                        "scene_confidence": 1.0
                    })

                finally:
                    # Clean up temp file
                    Path(temp_path).unlink(missing_ok=True)

            except Exception as e:
                logger.warning(f"Scene classification failed: {e}")
                results.append({
                    "primary_scene": "unknown",
                    "scene_scores": {},
                    "scene_confidence": 0.0
                })

        return results

    def _detect_objects_batch(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        """
        Batch object detection using NVIDIA VLM (implements abstract method).

        Args:
            images: List of PIL images

        Returns:
            List of detection dictionaries with keys:
            - objects_detected: List of object names
            - object_count: Total number of objects
            - dominant_object: Most prominent object (if any)
        """
        # Load models if needed
        if self._object_detector is None:
            self._load_nvidia_models()

        if self._object_detector == "FAILED":
            logger.warning("NVIDIA object detector not available, using fallback")
            return [{"objects_detected": [], "object_count": 0, "dominant_object": None}] * len(images)

        results = []
        for img in images:
            try:
                # Save image temporarily for API call
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    temp_path = tmp.name
                    img.save(temp_path, "JPEG")

                try:
                    # Detect objects
                    detection = self._object_detector.detect_products(temp_path)

                    results.append({
                        "objects_detected": detection.get("objects", []),
                        "object_count": detection.get("object_count", 0),
                        "dominant_object": detection.get("dominant_product"),
                    })

                finally:
                    # Clean up temp file
                    Path(temp_path).unlink(missing_ok=True)

            except Exception as e:
                logger.warning(f"Object detection failed: {e}")
                results.append({
                    "objects_detected": [],
                    "object_count": 0,
                    "dominant_object": None
                })

        return results

    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze single image with enhanced NVIDIA tagging.

        Args:
            image_path: Path to image file

        Returns:
            Analysis results with NVIDIA-generated tags
        """
        try:
            # Load image
            img = Image.open(image_path).convert('RGB')

            # Scene classification via NVIDIA VLM
            scene_result = self._classify_scene_batch([img])[0]

            # Object detection via NVIDIA VLM
            detection_result = self._detect_objects_batch([img])[0]

            # Merge results
            base_result = {**scene_result, **detection_result}

            # Add NVIDIA-specific enrichment: searchable tags
            if self._vlm_model and self._vlm_model != "FAILED":
                try:
                    tags = self._vlm_model.generate_searchable_tags(image_path)
                    base_result["ai_generated_tags"] = tags
                    logger.debug(f"Generated tags: {tags}")
                except Exception as e:
                    logger.warning(f"Tag generation failed: {e}")
                    base_result["ai_generated_tags"] = []

        except Exception as e:
            logger.error(f"Analysis failed for {image_path}: {e}")
            base_result = {
                "file_path": image_path,
                "error": str(e),
                "primary_scene": "unknown",
                "objects_detected": [],
                "object_count": 0
            }

        return base_result

    def analyze_batch(
        self,
        image_paths: List[str],
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Analyze batch of images with rate-limited API calls.

        Args:
            image_paths: List of image file paths
            show_progress: Show progress bar

        Returns:
            List of analysis results (one per image)
        """
        results = []
        iterator = tqdm(image_paths, desc="NVIDIA API analysis") if show_progress else image_paths

        for image_path in iterator:
            try:
                result = self.analyze_image(image_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Analysis failed for {image_path}: {e}")
                results.append({
                    "file_path": image_path,
                    "error": str(e),
                    "primary_scene": "unknown",
                    "objects_detected": [],
                    "object_count": 0,
                })

        return results
