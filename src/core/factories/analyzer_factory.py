"""Factory for creating image analyzers based on configuration.

Centralizes analyzer creation logic with priority-based selection:
1. NVIDIA Build API (cloud-based, highest quality tagging)
2. YOLO (fastest local object detection, 3-5x speedup)
3. TensorRT (2-4x speedup with FP16/INT8)
4. Standard PyTorch (baseline DETR)
"""

from typing import Optional
from ..utils.config import Config
from ..analyzers.metadata import MetadataExtractor
from ..analyzers.content import ContentAnalyzer
from ..interfaces.monitors import GPUMonitor


class AnalyzerFactory:
    """Factory for creating analyzers based on configuration."""

    def __init__(self, config: Config):
        """
        Initialize analyzer factory.

        Args:
            config: Configuration object
        """
        self.config = config

    def create_metadata_extractor(self) -> MetadataExtractor:
        """
        Create metadata extractor (always enabled).

        Returns:
            MetadataExtractor instance
        """
        return MetadataExtractor()

    def create_content_analyzer(self) -> Optional[ContentAnalyzer]:
        """
        Create content analyzer if enabled in config.

        Returns:
            ContentAnalyzer instance or None if disabled
        """
        if not self.config.get("analysis.content_analysis", True):
            return None

        num_workers = self.config.get("processing.max_workers", 4)
        return ContentAnalyzer(num_workers=num_workers)

    def create_ml_analyzer(
        self,
        gpu_monitor: Optional[GPUMonitor] = None
    ):
        """
        Create ML analyzer with priority-based selection.

        Priority order:
        1. NVIDIA Build API (if use_nvidia=true): NVIDIAVisionAnalyzer (cloud, highest quality)
        2. YOLO (if use_yolo=true): YOLOVisionAnalyzer (3-5x faster)
        3. TensorRT (if use_tensorrt=true): TensorRTVisionAnalyzer (2-4x faster)
        4. Standard: MLVisionAnalyzer (baseline DETR)

        Args:
            gpu_monitor: Optional GPU monitor for dependency injection

        Returns:
            ML analyzer instance or None if disabled
        """
        if not self.config.get("analysis.ml_analysis", True):
            return None

        # Get ML configuration
        ml_config = self.config.get("analysis.ml_models", {})
        if not isinstance(ml_config, dict):
            ml_config = {}

        # Common parameters
        common_params = {
            "use_gpu": ml_config.get("use_gpu", True),
            "scene_model": ml_config.get("scene_detection", "openai/clip-vit-base-patch32"),
            "min_confidence": self.config.get("tagging.min_confidence", 0.6),
            "gpu_monitor": gpu_monitor,
        }

        # Priority 1: NVIDIA Build API (highest quality)
        if ml_config.get("use_nvidia", False):
            return self._create_nvidia_analyzer(ml_config, common_params)

        # Priority 2: YOLO (fastest local)
        if ml_config.get("use_yolo", False):
            return self._create_yolo_analyzer(ml_config, common_params)

        # Priority 3: TensorRT (fast local)
        if ml_config.get("use_tensorrt", False):
            return self._create_tensorrt_analyzer(ml_config, common_params)

        # Priority 4: Standard DETR (baseline)
        return self._create_standard_analyzer(ml_config, common_params)

    def _create_nvidia_analyzer(self, ml_config: dict, common_params: dict):
        """
        Create NVIDIA Build API analyzer.

        Args:
            ml_config: ML models configuration
            common_params: Common analyzer parameters

        Returns:
            NVIDIAVisionAnalyzer instance
        """
        from ..analyzers.ml_vision_nvidia import NVIDIAVisionAnalyzer

        return NVIDIAVisionAnalyzer(
            batch_size=ml_config.get("batch_size", 8),
            model=ml_config.get("nvidia_model", "llama-vision"),
            api_key=ml_config.get("nvidia_api_key"),  # Optional, reads from env if None
            min_confidence=common_params["min_confidence"],
        )

    def _create_yolo_analyzer(self, ml_config: dict, common_params: dict):
        """
        Create YOLO-optimized analyzer.

        Args:
            ml_config: ML models configuration
            common_params: Common analyzer parameters

        Returns:
            YOLOVisionAnalyzer instance
        """
        from ..analyzers.ml_vision_yolo import YOLOVisionAnalyzer

        return YOLOVisionAnalyzer(
            batch_size=ml_config.get("batch_size", 16),
            yolo_model=ml_config.get("yolo_model", "yolov8n.pt"),
            precision=ml_config.get("precision", "fp16"),
            **common_params
        )

    def _create_tensorrt_analyzer(self, ml_config: dict, common_params: dict):
        """
        Create TensorRT-optimized analyzer.

        Args:
            ml_config: ML models configuration
            common_params: Common analyzer parameters

        Returns:
            TensorRTVisionAnalyzer instance
        """
        from ..analyzers.ml_vision_tensorrt import TensorRTVisionAnalyzer

        return TensorRTVisionAnalyzer(
            batch_size=ml_config.get("batch_size", 16),
            object_model=ml_config.get("object_detection", "facebook/detr-resnet-50"),
            use_tensorrt=True,
            precision=ml_config.get("tensorrt_precision", "fp16"),
            **common_params
        )

    def _create_standard_analyzer(self, ml_config: dict, common_params: dict):
        """
        Create standard PyTorch analyzer (baseline).

        Args:
            ml_config: ML models configuration
            common_params: Common analyzer parameters

        Returns:
            MLVisionAnalyzer instance
        """
        from ..analyzers.ml_vision import MLVisionAnalyzer

        return MLVisionAnalyzer(
            batch_size=ml_config.get("batch_size", 8),
            object_model=ml_config.get("object_detection", "facebook/detr-resnet-50"),
            **common_params
        )

    def create_all_analyzers(
        self,
        gpu_monitor: Optional[GPUMonitor] = None
    ) -> dict:
        """
        Create all enabled analyzers.

        Args:
            gpu_monitor: Optional GPU monitor for ML analyzer

        Returns:
            Dictionary with analyzer instances:
            {
                'metadata': MetadataExtractor,
                'content': ContentAnalyzer or None,
                'ml': ML analyzer or None
            }
        """
        return {
            'metadata': self.create_metadata_extractor(),
            'content': self.create_content_analyzer(),
            'ml': self.create_ml_analyzer(gpu_monitor=gpu_monitor),
        }
