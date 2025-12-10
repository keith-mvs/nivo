"""YOLOv8-optimized ML vision analysis for 3-5x faster object detection.

This module provides GPU-accelerated vision analysis with Ultralytics YOLOv8.
Combines CLIP for scene classification with YOLOv8 for object detection.

Key Features:
- YOLOv8 object detection (3-5x faster than DETR)
- CLIP scene classification (unchanged)
- FP16 precision with AMP for additional speedup
- Optimized batch processing
- GPU memory efficient
"""

import torch
from PIL import Image
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path
import warnings
from tqdm import tqdm
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.gpu_monitor import get_monitor

warnings.filterwarnings('ignore')


class YOLOVisionAnalyzer:
    """YOLOv8-optimized ML vision analyzer for maximum object detection performance."""

    def __init__(
        self,
        use_gpu: bool = True,
        batch_size: int = 16,
        scene_model: str = "openai/clip-vit-base-patch32",
        yolo_model: str = "yolov8n.pt",  # n=nano, s=small, m=medium, l=large, x=xlarge
        min_confidence: float = 0.6,
        precision: str = "fp16",
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
        """
        self.device = self._setup_device(use_gpu)
        self.batch_size = batch_size
        self.min_confidence = min_confidence
        self.precision = precision
        self.use_amp = precision == "fp16"

        # Model cache
        self._clip_model = None
        self._clip_processor = None
        self._yolo_model = None

        self.scene_model_name = scene_model
        self.yolo_model_name = yolo_model

        print(f"YOLO ML Analyzer initialized on device: {self.device}")
        print(f"Batch size: {batch_size}")
        print(f"Precision: {precision} (AMP: {self.use_amp})")
        print(f"YOLO variant: {yolo_model}")

    def _setup_device(self, use_gpu: bool) -> torch.device:
        """Setup computation device."""
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
        """Load CLIP model for scene classification."""
        try:
            from transformers import CLIPProcessor, CLIPModel

            print(f"Loading CLIP model: {self.scene_model_name}")

            try:
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

    def _classify_scene_batch(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        """Classify scenes using CLIP with AMP optimization."""
        results = []

        try:
            if self._clip_model is None:
                self._load_clip_model()

            if self._clip_model == "FAILED":
                return [{"primary_scene": "unknown", "scene_scores": {}} for _ in images]

            # Scene categories
            scene_labels = [
                "indoor", "outdoor", "nature", "city", "landscape",
                "portrait", "food", "animal", "vehicle", "architecture",
                "beach", "mountain", "forest", "sunset", "night",
                "party", "sports", "concert", "travel", "selfie"
            ]

            # Process batch
            inputs = self._clip_processor(
                text=scene_labels,
                images=images,
                return_tensors="pt",
                padding=True
            ).to(self.device)

            model = self._clip_model

            with torch.no_grad():
                # Enable AMP for FP16 if configured
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    outputs = model(**inputs)
                    logits = outputs.logits_per_image
                    probs = logits.softmax(dim=1).cpu().numpy()

            # Extract results for each image
            for prob in probs:
                scene_scores = {label: float(score) for label, score in zip(scene_labels, prob)}
                primary_scene = max(scene_scores, key=scene_scores.get)
                results.append({
                    "primary_scene": primary_scene,
                    "scene_scores": scene_scores,
                    "scene_confidence": scene_scores[primary_scene],
                })

        except Exception as e:
            results = [{"scene_error": str(e)} for _ in images]

        return results

    def _detect_objects_batch(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        """Batch object detection using YOLOv8."""
        results = []

        try:
            if self._yolo_model is None:
                self._load_yolo_model()

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

    def analyze(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze single image.

        Args:
            image_path: Path to image file

        Returns:
            Analysis results with scenes and objects
        """
        try:
            image = Image.open(image_path).convert("RGB")

            # Classify scene
            scene_result = self._classify_scene_batch([image])[0]

            # Detect objects
            object_result = self._detect_objects_batch([image])[0]

            # Combine results
            return {**scene_result, **object_result}

        except Exception as e:
            return {"error": str(e)}

    def analyze_batch(
        self,
        image_paths: List[str],
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Batch analyze images with GPU optimization.

        Args:
            image_paths: List of image file paths
            show_progress: Show progress bar

        Returns:
            List of analysis results
        """
        results = []
        monitor = get_monitor()

        # Start GPU monitoring
        if self.device.type == "cuda":
            monitor.start()

        # Process in batches
        total_batches = (len(image_paths) + self.batch_size - 1) // self.batch_size

        with tqdm(total=len(image_paths), desc="YOLO ML Analysis", unit="img",
                  disable=not show_progress, ncols=120,
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}') as pbar:

            for i in range(0, len(image_paths), self.batch_size):
                batch_paths = image_paths[i:i + self.batch_size]

                # Load images
                batch_images = []
                for path in batch_paths:
                    try:
                        img = Image.open(path).convert("RGB")
                        batch_images.append(img)
                    except Exception as e:
                        batch_images.append(None)

                # Process valid images
                valid_images = [img for img in batch_images if img is not None]

                if valid_images:
                    # Scene classification
                    scene_results = self._classify_scene_batch(valid_images)

                    # Object detection
                    object_results = self._detect_objects_batch(valid_images)

                    # Combine results
                    valid_idx = 0
                    for img in batch_images:
                        if img is not None:
                            combined = {**scene_results[valid_idx], **object_results[valid_idx]}
                            results.append(combined)
                            valid_idx += 1
                        else:
                            results.append({"error": "Failed to load image"})

                # Update progress bar
                status_parts = []
                if results and "primary_scene" in results[-1]:
                    status_parts.append(f"Scene: {results[-1]['primary_scene']}")
                if self.device.type == "cuda":
                    status_parts.append(monitor.get_status_string())

                pbar.set_postfix_str(" | ".join(status_parts))
                pbar.update(len(batch_paths))

        # Stop monitoring and show stats
        if self.device.type == "cuda":
            print()
            monitor.print_stats()
            monitor.stop()

        return results

    def clear_cache(self):
        """Clear GPU memory cache."""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def get_memory_usage(self) -> Optional[Dict[str, float]]:
        """Get current GPU memory usage."""
        if self.device.type == "cuda":
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9

            return {
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "total_gb": total,
                "utilization": (allocated / total) * 100,
            }
        return None
