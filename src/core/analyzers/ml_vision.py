"""ML-powered vision analysis using PyTorch with GPU acceleration.

Supports:
- Scene classification (indoor, outdoor, landscape, etc.) - inherited from BaseMLAnalyzer
- Object detection using DETR (facebook/detr-resnet-50)
- Semantic understanding for intelligent tagging
"""

import torch
from PIL import Image
from typing import Dict, Any, List, Optional
from tqdm import tqdm

from .base_ml_analyzer import BaseMLAnalyzer
from ..interfaces.monitors import GPUMonitor
from ..utils.logging_config import get_logger



logger = get_logger(__name__)
class MLVisionAnalyzer(BaseMLAnalyzer):
    """ML-powered image analysis with GPU acceleration using DETR for object detection."""

    def __init__(
        self,
        use_gpu: bool = True,
        batch_size: int = 8,
        scene_model: str = "openai/clip-vit-base-patch32",
        object_model: str = "facebook/detr-resnet-50",
        min_confidence: float = 0.6,
        gpu_monitor: Optional[GPUMonitor] = None,
    ):
        """
        Initialize ML vision analyzer with DETR.

        Args:
            use_gpu: Enable GPU acceleration if available
            batch_size: Batch size for processing
            scene_model: HuggingFace model for scene classification
            object_model: HuggingFace model for object detection (DETR)
            min_confidence: Minimum confidence threshold for predictions
            gpu_monitor: Optional GPU monitor for tracking (defaults to global instance)
        """
        # Initialize base class (handles device, CLIP, monitoring)
        super().__init__(
            use_gpu=use_gpu,
            batch_size=batch_size,
            scene_model=scene_model,
            min_confidence=min_confidence,
            gpu_monitor=gpu_monitor,
        )

        # DETR-specific configuration
        self.object_model_name = object_model

        # DETR model cache
        self._detr_model = None
        self._detr_processor = None

        logger.info(f"ML Analyzer initialized on device: {self.device}")

    def _load_detection_model(self):
        """Load DETR model for object detection (implements abstract method)."""
        self._load_detr_model()

    def _load_detr_model(self):
        """Load DETR model for object detection."""
        try:
            from transformers import DetrImageProcessor, DetrForObjectDetection

            logger.debug(f"Loading DETR model: {self.object_model_name}")
            self._detr_processor = DetrImageProcessor.from_pretrained(self.object_model_name)
            self._detr_model = DetrForObjectDetection.from_pretrained(self.object_model_name).to(self.device)
            self._detr_model.eval()
            logger.debug("DETR model loaded successfully")

        except Exception as e:
            logger.error(f"ERROR: Failed to load DETR model: {e}")
            self._detr_model = "FAILED"
            self._detr_processor = None

    def _detect_objects_batch(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        """
        Batch object detection using DETR (implements abstract method).

        Note: DETR doesn't batch well due to varying image sizes,
        so we process one image at a time.

        Args:
            images: List of PIL images

        Returns:
            List of detection results with objects, counts, etc.
        """
        results = []

        try:
            # Lazy load DETR model
            if self._detr_model is None:
                self._load_detr_model()

            # Graceful degradation if loading failed
            if self._detr_model == "FAILED":
                return [{"object_error": "DETR model failed to load"} for _ in images]

            # Process each image individually (DETR limitation)
            for image in images:
                result = self._detect_objects_single(image)
                results.append(result)

        except Exception as e:
            results = [{"object_error": str(e)} for _ in images]

        return results

    def _detect_objects_single(self, image: Image.Image) -> Dict[str, Any]:
        """
        Detect objects in single image using DETR.

        Args:
            image: PIL Image

        Returns:
            Detection results dictionary
        """
        try:
            # Prepare inputs
            inputs = self._detr_processor(images=image, return_tensors="pt").to(self.device)

            # Get predictions
            with torch.no_grad():
                outputs = self._detr_model(**inputs)

            # Process outputs
            target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
            results_processed = self._detr_processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=self.min_confidence
            )[0]

            # Extract objects
            objects = []
            object_counts = {}

            for score, label in zip(results_processed["scores"], results_processed["labels"]):
                label_name = self._detr_model.config.id2label[label.item()]
                score_value = float(score)

                if score_value >= self.min_confidence:
                    objects.append({
                        "object": label_name,
                        "confidence": score_value,
                    })

                    # Count objects
                    object_counts[label_name] = object_counts.get(label_name, 0) + 1

            return {
                "objects": objects,
                "object_count": len(objects),
                "unique_objects": list(object_counts.keys()),
                "object_counts": object_counts,
            }

        except Exception as e:
            return {"object_error": str(e)}

    def analyze_batch(self, image_paths: List[str], show_progress: bool = True) -> List[Dict[str, Any]]:
        """
        Analyze multiple images in batches for GPU efficiency.

        Args:
            image_paths: List of image file paths
            show_progress: Show progress bar

        Returns:
            List of analysis results
        """
        results = []
        monitor = self.gpu_monitor

        # Start GPU monitoring
        if self.device.type == "cuda":
            monitor.start()

        # Calculate total batches
        total_batches = (len(image_paths) + self.batch_size - 1) // self.batch_size

        # Process in batches with progress bar
        iterator = range(0, len(image_paths), self.batch_size)
        if show_progress:
            iterator = tqdm(
                iterator,
                total=total_batches,
                desc="ML Analysis (DETR)",
                unit="batch",
                ncols=120,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}'
            )

        for i in iterator:
            batch_paths = image_paths[i:i + self.batch_size]
            batch_images = []

            # Load batch of images
            for path in batch_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    batch_images.append(img)
                except Exception as e:
                    results.append({"error": f"Failed to load {path}: {e}"})

            if not batch_images:
                continue

            # Scene classification (uses base class method)
            scene_results = self._classify_scene_batch(batch_images, use_amp=False)

            # Object detection (DETR-specific)
            object_results = self._detect_objects_batch(batch_images)

            # Merge results
            for scene_res, obj_res in zip(scene_results, object_results):
                combined = {**scene_res, **obj_res}

                # Generate tags from scene and objects
                tags = self._generate_tags(combined)
                combined["tags"] = tags
                combined["tag_string"] = ", ".join(tags)

                results.append(combined)

            # Show GPU stats in progress bar
            if show_progress and results:
                status_parts = []

                # Add scene info
                last_result = results[-1]
                if "primary_scene" in last_result:
                    status_parts.append(f"Scene: {last_result['primary_scene']}")

                # Add memory info
                if self.device.type == "cuda":
                    mem = self.get_memory_usage()
                    if mem:
                        status_parts.append(f"GPU: {mem['allocated_gb']:.1f}GB")

                if status_parts:
                    iterator.set_postfix_str(" | ".join(status_parts))

        # Stop GPU monitoring
        if self.device.type == "cuda":
            monitor.stop()

        return results

    def _generate_tags(self, analysis_result: Dict[str, Any]) -> List[str]:
        """
        Generate semantic tags from analysis results.

        Args:
            analysis_result: Combined scene and object detection results

        Returns:
            List of tags
        """
        tags = []

        # Add primary scene
        if "primary_scene" in analysis_result:
            tags.append(analysis_result["primary_scene"])

        # Add detected objects
        if "unique_objects" in analysis_result:
            tags.extend(analysis_result["unique_objects"][:5])  # Limit to top 5

        return tags


# Backwards compatibility: keep old name as alias
MLVisionAnalyzer_Legacy = MLVisionAnalyzer
