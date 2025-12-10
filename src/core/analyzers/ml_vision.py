"""ML-powered vision analysis using PyTorch with GPU acceleration.

Supports:
- Scene classification (indoor, outdoor, landscape, etc.)
- Object detection (people, animals, vehicles, etc.)
- Semantic understanding for intelligent tagging
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import warnings
from tqdm import tqdm
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.gpu_monitor import get_monitor

# Suppress warnings
warnings.filterwarnings('ignore')


class MLVisionAnalyzer:
    """ML-powered image analysis with GPU acceleration."""

    def __init__(
        self,
        use_gpu: bool = True,
        batch_size: int = 8,
        scene_model: str = "openai/clip-vit-base-patch32",
        object_model: str = "facebook/detr-resnet-50",
        min_confidence: float = 0.6,
    ):
        """
        Initialize ML vision analyzer.

        Args:
            use_gpu: Enable GPU acceleration if available
            batch_size: Batch size for processing
            scene_model: HuggingFace model for scene classification
            object_model: HuggingFace model for object detection
            min_confidence: Minimum confidence threshold for predictions
        """
        self.device = self._setup_device(use_gpu)
        self.batch_size = batch_size
        self.min_confidence = min_confidence

        # Model cache
        self._clip_model = None
        self._clip_processor = None
        self._detr_model = None
        self._detr_processor = None

        self.scene_model_name = scene_model
        self.object_model_name = object_model

        print(f"ML Analyzer initialized on device: {self.device}")

    def _setup_device(self, use_gpu: bool) -> torch.device:
        """Setup computation device (GPU/CPU)."""
        if use_gpu and torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            device = torch.device("cpu")
            if use_gpu:
                print("GPU requested but not available, using CPU")

        return device

    def analyze(self, image_path: str) -> Dict[str, Any]:
        """
        Perform ML-powered analysis on single image.

        Args:
            image_path: Path to image file

        Returns:
            Analysis results with scenes, objects, and tags
        """
        results = {
            "image_path": image_path,
            "device_used": str(self.device),
        }

        try:
            img = Image.open(image_path).convert("RGB")

            # Scene classification using CLIP
            scene_results = self._classify_scene(img)
            results.update(scene_results)

            # Object detection using DETR
            object_results = self._detect_objects(img)
            results.update(object_results)

            # Generate comprehensive tags
            tags = self._generate_tags(scene_results, object_results)
            results["tags"] = tags
            results["tag_string"] = ", ".join(tags)

        except Exception as e:
            results["error"] = f"ML analysis failed: {e}"

        return results

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

        # Start GPU monitoring
        monitor = get_monitor()
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
                desc="ML Analysis",
                unit="batch",
                ncols=120,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}'
            )

        for i in iterator:
            batch_paths = image_paths[i:i + self.batch_size]
            batch_results = self._analyze_batch_internal(batch_paths)
            results.extend(batch_results)

            # Show GPU stats and scene info
            if show_progress and batch_results:
                status_parts = []

                # Add scene info
                first_result = batch_results[0]
                if "primary_scene" in first_result:
                    status_parts.append(f"Scene: {first_result['primary_scene']}")

                # Add GPU stats
                if self.device.type == "cuda":
                    gpu_status = monitor.get_status_string()
                    status_parts.append(gpu_status)

                iterator.set_postfix_str(" | ".join(status_parts))

            # Clear GPU cache
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        # Print final GPU stats
        if self.device.type == "cuda" and show_progress:
            print()  # New line after progress bar
            monitor.print_stats()
            monitor.stop()

        return results

    def _analyze_batch_internal(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """Internal batch processing."""
        results = []

        try:
            # Load images
            images = []
            valid_paths = []
            for path in image_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    images.append(img)
                    valid_paths.append(path)
                except Exception as e:
                    results.append({"image_path": path, "error": str(e)})

            if not images:
                return results

            # Batch scene classification
            scene_results = self._classify_scene_batch(images)

            # Batch object detection
            object_results = self._detect_objects_batch(images)

            # Combine results
            for path, scene, objects in zip(valid_paths, scene_results, object_results):
                result = {
                    "image_path": path,
                    "device_used": str(self.device),
                }
                result.update(scene)
                result.update(objects)

                tags = self._generate_tags(scene, objects)
                result["tags"] = tags
                result["tag_string"] = ", ".join(tags)

                results.append(result)

        except Exception as e:
            for path in image_paths:
                results.append({"image_path": path, "error": str(e)})

        return results

    def _classify_scene(self, image: Image.Image) -> Dict[str, Any]:
        """Classify scene using CLIP zero-shot classification."""
        try:
            # Load CLIP model
            if self._clip_model is None:
                self._load_clip_model()

            # If CLIP failed to load, return empty result
            if self._clip_model == "FAILED":
                return {"primary_scene": "unknown", "scene_confidence": 0.0}

            # Scene categories
            scene_categories = [
                "indoor scene", "outdoor scene", "landscape", "cityscape",
                "portrait photo", "group photo", "nature photo", "architecture",
                "food photo", "product photo", "screenshot", "document",
                "night photo", "sunset", "beach", "mountain", "forest",
                "street photo", "aerial view", "close-up", "macro photo"
            ]

            # Prepare inputs
            from transformers import CLIPProcessor, CLIPModel

            inputs = self._clip_processor(
                text=scene_categories,
                images=image,
                return_tensors="pt",
                padding=True
            ).to(self.device)

            # Get predictions
            with torch.no_grad():
                outputs = self._clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]

            # Get top scenes
            top_indices = np.argsort(probs)[::-1][:5]
            scenes = []
            for idx in top_indices:
                if probs[idx] >= self.min_confidence:
                    scenes.append({
                        "scene": scene_categories[idx],
                        "confidence": float(probs[idx]),
                    })

            primary_scene = scenes[0]["scene"] if scenes else "unknown"

            return {
                "primary_scene": primary_scene,
                "all_scenes": scenes,
                "scene_confidence": scenes[0]["confidence"] if scenes else 0.0,
            }

        except Exception as e:
            return {"scene_error": str(e)}

    def _classify_scene_batch(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        """Batch scene classification."""
        results = []

        try:
            if self._clip_model is None:
                self._load_clip_model()

            # If CLIP failed to load, return empty results
            if self._clip_model == "FAILED":
                return [{"primary_scene": "unknown", "scene_confidence": 0.0} for _ in images]

            scene_categories = [
                "indoor scene", "outdoor scene", "landscape", "cityscape",
                "portrait photo", "nature photo", "food photo", "night photo"
            ]

            from transformers import CLIPProcessor

            # Process batch
            inputs = self._clip_processor(
                text=scene_categories,
                images=images,
                return_tensors="pt",
                padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self._clip_model(**inputs)
                logits = outputs.logits_per_image
                probs = logits.softmax(dim=1).cpu().numpy()

            # Parse results
            for prob in probs:
                top_idx = np.argmax(prob)
                results.append({
                    "primary_scene": scene_categories[top_idx],
                    "scene_confidence": float(prob[top_idx]),
                })

        except Exception as e:
            results = [{"scene_error": str(e)} for _ in images]

        return results

    def _detect_objects(self, image: Image.Image) -> Dict[str, Any]:
        """Detect objects using DETR."""
        try:
            if self._detr_model is None:
                self._load_detr_model()

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

    def _detect_objects_batch(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        """Batch object detection."""
        results = []

        try:
            if self._detr_model is None:
                self._load_detr_model()

            # Process each image (DETR batch processing is complex)
            for image in images:
                result = self._detect_objects(image)
                results.append(result)

        except Exception as e:
            results = [{"object_error": str(e)} for _ in images]

        return results

    def _load_clip_model(self):
        """Lazy load CLIP model with safetensors fallback."""
        try:
            from transformers import CLIPProcessor, CLIPModel

            print(f"Loading CLIP model: {self.scene_model_name}")

            # Try loading with safetensors first (more secure)
            try:
                self._clip_processor = CLIPProcessor.from_pretrained(self.scene_model_name)
                self._clip_model = CLIPModel.from_pretrained(
                    self.scene_model_name,
                    use_safetensors=True  # Force safetensors format
                ).to(self.device)
                self._clip_model.eval()
                print("CLIP model loaded successfully (safetensors)")
            except Exception as e:
                # If safetensors fails, try regular loading
                print(f"Safetensors load failed, trying regular load: {e}")
                self._clip_processor = CLIPProcessor.from_pretrained(self.scene_model_name)
                self._clip_model = CLIPModel.from_pretrained(self.scene_model_name).to(self.device)
                self._clip_model.eval()
                print("CLIP model loaded successfully")

        except Exception as e:
            print(f"Failed to load CLIP model: {e}")
            print("Scene classification will be disabled. Only object detection will be used.")
            # Set to a sentinel value to indicate loading was attempted but failed
            self._clip_model = "FAILED"
            self._clip_processor = None

    def _load_detr_model(self):
        """Lazy load DETR model."""
        try:
            from transformers import DetrImageProcessor, DetrForObjectDetection

            print(f"Loading DETR model: {self.object_model_name}")
            self._detr_processor = DetrImageProcessor.from_pretrained(self.object_model_name)
            self._detr_model = DetrForObjectDetection.from_pretrained(self.object_model_name).to(self.device)
            self._detr_model.eval()
            print("DETR model loaded successfully")

        except Exception as e:
            print(f"Failed to load DETR model: {e}")
            raise

    def _generate_tags(self, scene_data: Dict, object_data: Dict) -> List[str]:
        """Generate comprehensive tags from analysis results."""
        tags = set()

        # Add primary scene (if available and not unknown)
        if "primary_scene" in scene_data and scene_data["primary_scene"] != "unknown":
            scene = scene_data["primary_scene"].replace(" scene", "").replace(" photo", "")
            tags.add(scene)

        # Add unique objects
        if "unique_objects" in object_data:
            for obj in object_data["unique_objects"][:10]:  # Limit to top 10
                tags.add(obj)

        # If no tags were generated, add a generic tag based on objects
        if not tags and "object_count" in object_data and object_data["object_count"] > 0:
            tags.add("photo")

        return sorted(list(tags))

    def get_memory_usage(self) -> Dict[str, float]:
        """Get GPU/CPU memory usage."""
        memory_info = {}

        if self.device.type == "cuda":
            memory_info["gpu_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
            memory_info["gpu_reserved_gb"] = torch.cuda.memory_reserved() / 1e9
            memory_info["gpu_max_allocated_gb"] = torch.cuda.max_memory_allocated() / 1e9

        return memory_info

    def clear_cache(self):
        """Clear GPU cache to free memory."""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
