"""YOLOv8-optimized ML vision analysis for 3-5x faster object detection.

This module provides GPU-accelerated vision analysis with Ultralytics YOLOv8.
Combines CLIP for scene classification (from base class) with YOLOv8 for object detection.

Key Features:
- YOLOv8 object detection (3-5x faster than DETR)
- CLIP scene classification (inherited from BaseMLAnalyzer)
- FP16 precision with AMP for additional speedup
- Optimized batch processing
- GPU memory efficient
"""

import torch
from PIL import Image
import numpy as np
from typing import Dict, Any, List, Optional
from tqdm import tqdm

from .base_ml_analyzer import BaseMLAnalyzer
from ..interfaces.monitors import GPUMonitor


class YOLOVisionAnalyzer(BaseMLAnalyzer):
    """YOLOv8-optimized ML vision analyzer for maximum object detection performance."""

    def __init__(
        self,
        use_gpu: bool = True,
        batch_size: int = 16,
        scene_model: str = "openai/clip-vit-base-patch32",
        yolo_model: str = "yolov8n.pt",  # n=nano, s=small, m=medium, l=large, x=xlarge
        min_confidence: float = 0.6,
        precision: str = "fp16",
        gpu_monitor: Optional[GPUMonitor] = None,
    ):
        """
        Initialize YOLO-optimized vision analyzer.

        Args:
            use_gpu: Enable GPU acceleration
            batch_size: Batch size for processing
            scene_model: CLIP model for scene classification
            yolo_model: YOLO model variant (yolov8n.pt to yolov8x.pt)
            min_confidence: Confidence threshold for detections
            precision: Precision mode ("fp16" or "fp32")
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

        # YOLO-specific configuration
        self.precision = precision
        self.use_amp = precision == "fp16"
        self.yolo_model_name = yolo_model

        # YOLO model cache
        self._yolo_model = None

        print(f"YOLO ML Analyzer initialized on device: {self.device}")
        print(f"Batch size: {batch_size}")
        print(f"Precision: {precision} (AMP: {self.use_amp})")
        print(f"YOLO variant: {yolo_model}")

    def _load_detection_model(self):
        """Load YOLOv8 model for object detection (implements abstract method)."""
        self._load_yolo_model()

    def _load_yolo_model(self):
        """Load YOLOv8 model for object detection."""
        try:
            from ultralytics import YOLO

            print(f"Loading YOLO model: {self.yolo_model_name}")

            # Load model and set device
            self._yolo_model = YOLO(self.yolo_model_name)

            # Configure for GPU if available
            if self.device.type == "cuda":
                self._yolo_model.to(self.device)

            print(f"YOLOv8 model loaded successfully on {self.device}")

        except Exception as e:
            print(f"ERROR: Failed to load YOLO model: {e}")
            self._yolo_model = "FAILED"

    def _detect_objects_batch(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        """
        Batch object detection using YOLOv8 (implements abstract method).

        Args:
            images: List of PIL images

        Returns:
            List of detection results with objects, counts, etc.
        """
        results = []

        try:
            # Lazy load YOLO model
            if self._yolo_model is None:
                self._load_yolo_model()

            # Graceful degradation if loading failed
            if self._yolo_model == "FAILED":
                return [{"object_error": "YOLO model failed to load"} for _ in images]

            # YOLOv8 native batch inference
            # Convert PIL images to numpy arrays
            img_arrays = [np.array(img) for img in images]

            # Run inference with AMP and batch processing
            with torch.no_grad():
                if self.use_amp and self.device.type == "cuda":
                    with torch.cuda.amp.autocast():
                        yolo_results = self._yolo_model(
                            img_arrays,
                            conf=self.min_confidence,
                            verbose=False,
                            device=self.device
                        )
                else:
                    yolo_results = self._yolo_model(
                        img_arrays,
                        conf=self.min_confidence,
                        verbose=False,
                        device=self.device
                    )

            # Process results for each image
            for yolo_result in yolo_results:
                objects = []
                object_counts = {}

                # Extract detections
                boxes = yolo_result.boxes
                for box in boxes:
                    # Get class name and confidence
                    cls_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    label_name = yolo_result.names[cls_id]

                    if confidence >= self.min_confidence:
                        objects.append({
                            "object": label_name,
                            "confidence": confidence,
                        })
                        object_counts[label_name] = object_counts.get(label_name, 0) + 1

                results.append({
                    "objects": objects,
                    "object_count": len(objects),
                    "unique_objects": list(object_counts.keys()),
                    "object_counts": object_counts,
                })

        except Exception as e:
            results = [{"object_error": str(e)} for _ in images]

        return results

    def analyze_batch(
        self,
        image_paths: List[str],
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Batch analyze images with YOLO-optimized GPU processing.

        Args:
            image_paths: List of image file paths
            show_progress: Show progress bar

        Returns:
            List of analysis results

        Overrides base class to provide YOLO-specific batch optimization.
        """
        results = []
        monitor = self.gpu_monitor

        # Start GPU monitoring
        if self.device.type == "cuda":
            monitor.start()

        # Process in batches
        total_batches = (len(image_paths) + self.batch_size - 1) // self.batch_size

        iterator = range(0, len(image_paths), self.batch_size)
        if show_progress:
            iterator = tqdm(
                iterator,
                total=total_batches,
                desc="Analyzing images (YOLO)",
                unit="batch"
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

            # Scene classification (uses base class method with AMP)
            scene_results = self._classify_scene_batch(batch_images, use_amp=self.use_amp)

            # Object detection (YOLO-specific)
            object_results = self._detect_objects_batch(batch_images)

            # Merge results
            for scene_res, obj_res in zip(scene_results, object_results):
                combined = {**scene_res, **obj_res}
                results.append(combined)

        # Stop GPU monitoring
        if self.device.type == "cuda":
            monitor.stop()

        return results


# Backwards compatibility: keep old name as alias
YOLOVisionAnalyzer_Legacy = YOLOVisionAnalyzer
