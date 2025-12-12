"""Base class for ML vision analyzers with shared CLIP and common functionality.

This base class extracts ~200 lines of duplicated code from YOLO, DETR, and TensorRT analyzers.
Uses Template Method pattern for object detection while sharing scene classification logic.
"""

import torch
from PIL import Image
import numpy as np
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
import warnings

from ..interfaces.analyzers import MLAnalyzer
from ..interfaces.monitors import GPUMonitor
from ..utils.gpu_monitor import get_monitor

warnings.filterwarnings('ignore')


class BaseMLAnalyzer(MLAnalyzer, ABC):
    """
    Abstract base class for ML vision analyzers.

    Provides shared functionality:
    - Device setup (GPU/CPU)
    - CLIP model loading and scene classification
    - Memory management
    - Batch processing orchestration

    Subclasses must implement:
    - _load_detection_model(): Load object detection model
    - _detect_objects_batch(): Run object detection
    """

    def __init__(
        self,
        use_gpu: bool = True,
        batch_size: int = 8,
        scene_model: str = "openai/clip-vit-base-patch32",
        min_confidence: float = 0.6,
        gpu_monitor: Optional[GPUMonitor] = None,
    ):
        """
        Initialize base ML analyzer.

        Args:
            use_gpu: Enable GPU acceleration
            batch_size: Batch size for processing
            scene_model: CLIP model for scene classification
            min_confidence: Minimum confidence threshold
            gpu_monitor: Optional GPU monitor (defaults to global instance)
        """
        self.device = self._setup_device(use_gpu)
        self.batch_size = batch_size
        self.min_confidence = min_confidence
        self.gpu_monitor = gpu_monitor or get_monitor()

        # CLIP model cache (shared across all subclasses)
        self._clip_model = None
        self._clip_processor = None
        self.scene_model_name = scene_model

        # Scene labels for classification
        self.scene_labels = [
            "indoor", "outdoor", "nature", "city", "landscape",
            "portrait", "food", "animal", "vehicle", "architecture",
            "beach", "mountain", "forest", "sunset", "night",
            "party", "sports", "concert", "travel", "selfie"
        ]

    def _setup_device(self, use_gpu: bool) -> torch.device:
        """
        Setup computation device (GPU or CPU).

        Args:
            use_gpu: Whether to use GPU if available

        Returns:
            torch.device configured for inference
        """
        if use_gpu and torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            device = torch.device("cpu")
            if use_gpu:
                print("WARNING: GPU requested but not available. Using CPU.")
            else:
                print("Using CPU for inference")
        return device

    def _load_clip_model(self):
        """
        Load CLIP model for scene classification.

        Uses safetensors format first, falls back to regular loading.
        Sets self._clip_model to "FAILED" sentinel on error for graceful degradation.
        """
        try:
            from transformers import CLIPProcessor, CLIPModel

            print(f"Loading CLIP model: {self.scene_model_name}")

            try:
                # Try safetensors first (PyTorch 2.5.1+ security requirement)
                self._clip_processor = CLIPProcessor.from_pretrained(self.scene_model_name)
                self._clip_model = CLIPModel.from_pretrained(
                    self.scene_model_name,
                    use_safetensors=True
                ).to(self.device)
                self._clip_model.eval()
                print("CLIP model loaded successfully (safetensors)")

            except Exception as e:
                # Fallback without safetensors
                print(f"Safetensors load failed, trying regular loading: {e}")
                self._clip_processor = CLIPProcessor.from_pretrained(self.scene_model_name)
                self._clip_model = CLIPModel.from_pretrained(self.scene_model_name).to(self.device)
                self._clip_model.eval()
                print("CLIP model loaded successfully (regular)")

        except Exception as e:
            print(f"ERROR: Failed to load CLIP model: {e}")
            self._clip_model = "FAILED"

    def _classify_scene_batch(
        self,
        images: List[Image.Image],
        use_amp: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Classify scenes using CLIP model.

        Args:
            images: List of PIL images
            use_amp: Enable automatic mixed precision (FP16)

        Returns:
            List of scene classification results
        """
        results = []

        try:
            # Lazy load CLIP model
            if self._clip_model is None:
                self._load_clip_model()

            # Graceful degradation if loading failed
            if self._clip_model == "FAILED":
                return [{"primary_scene": "unknown", "scene_scores": {}} for _ in images]

            # Process batch with CLIP
            inputs = self._clip_processor(
                text=self.scene_labels,
                images=images,
                return_tensors="pt",
                padding=True
            ).to(self.device)

            with torch.no_grad():
                # Enable AMP for FP16 if requested
                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = self._clip_model(**inputs)
                    logits = outputs.logits_per_image
                    probs = logits.softmax(dim=1).cpu().numpy()

            # Extract results for each image
            for prob in probs:
                scene_scores = {label: float(score) for label, score in zip(self.scene_labels, prob)}
                primary_scene = max(scene_scores, key=scene_scores.get)
                results.append({
                    "primary_scene": primary_scene,
                    "scene_scores": scene_scores,
                    "scene_confidence": scene_scores[primary_scene],
                })

        except Exception as e:
            results = [{"scene_error": str(e)} for _ in images]

        return results

    @abstractmethod
    def _load_detection_model(self):
        """
        Load object detection model (YOLO, DETR, or TensorRT).

        Must be implemented by subclasses.
        Should set model to "FAILED" sentinel on error.
        """
        pass

    @abstractmethod
    def _detect_objects_batch(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        """
        Detect objects in batch of images.

        Args:
            images: List of PIL images

        Returns:
            List of detection results with objects, counts, etc.

        Must be implemented by subclasses (YOLO, DETR, or TensorRT specific logic).
        """
        pass

    def analyze(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze single image (delegates to batch method).

        Args:
            image_path: Path to image file

        Returns:
            Analysis results dictionary
        """
        results = self.analyze_batch([image_path], show_progress=False)
        return results[0] if results else {}

    def analyze_batch(self, image_paths: List[str], show_progress: bool = True) -> List[Dict[str, Any]]:
        """
        Analyze batch of images with scene classification and object detection.

        Args:
            image_paths: List of image file paths
            show_progress: Show progress bar

        Returns:
            List of analysis results

        Template method that coordinates scene classification and object detection.
        Subclasses provide detection implementation via _detect_objects_batch().
        """
        results = []
        monitor = self.gpu_monitor

        # Start GPU monitoring
        if self.device.type == "cuda":
            monitor.start()

        # Process images
        for image_path in image_paths:
            try:
                img = Image.open(image_path).convert('RGB')

                # Scene classification
                scene_result = self._classify_scene_batch([img], use_amp=False)[0]

                # Object detection (subclass-specific)
                detection_result = self._detect_objects_batch([img])[0]

                # Merge results
                result = {**scene_result, **detection_result}
                results.append(result)

            except Exception as e:
                results.append({"error": str(e)})

        # Stop GPU monitoring
        if self.device.type == "cuda":
            monitor.stop()

        return results

    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current GPU memory usage.

        Returns:
            Dictionary with memory statistics (or empty dict if CPU)
        """
        if self.device.type == "cuda":
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9

            return {
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "total_gb": total,
                "utilization_percent": (allocated / total) * 100,
            }
        return {}

    def clear_cache(self):
        """Clear GPU cache and free memory."""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            print("GPU cache cleared")

    def get_device_info(self) -> Dict[str, Any]:
        """
        Get device information.

        Returns:
            Dictionary with device details
        """
        info = {
            "device_type": self.device.type,
            "device": str(self.device),
        }

        if self.device.type == "cuda":
            info.update({
                "device_name": torch.cuda.get_device_name(0),
                "device_count": torch.cuda.device_count(),
                "cuda_version": torch.version.cuda,
                "pytorch_version": torch.__version__,
            })

        return info
