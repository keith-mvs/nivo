"""Configuration models with validation using Pydantic."""

from typing import List, Dict, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


class MLModelsConfig(BaseModel):
    """ML models configuration."""

    model_config = ConfigDict(extra="allow")  # Allow additional fields for backwards compatibility

    scene_detection: str = "openai/clip-vit-base-patch32"
    object_detection: str = "facebook/detr-resnet-50"
    use_gpu: bool = True
    batch_size: int = Field(default=8, gt=0, le=64)
    use_yolo: bool = False
    yolo_model: str = "yolov8n.pt"
    min_confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    precision: str = Field(default="fp16", pattern="^(fp16|fp32)$")


class QualityConfig(BaseModel):
    """Quality threshold configuration."""

    min_resolution: List[int] = Field(default=[640, 480], min_length=2, max_length=2)
    blur_threshold: float = Field(default=100.0, ge=0.0)
    jpeg_quality_threshold: int = Field(default=85, ge=0, le=100)

    @field_validator('min_resolution')
    @classmethod
    def validate_resolution(cls, v):
        """Validate resolution is [width, height]."""
        if len(v) != 2 or any(x <= 0 for x in v):
            raise ValueError("Resolution must be [width, height] with positive values")
        return v


class AnalysisConfig(BaseModel):
    """Analysis phase configuration."""

    extract_metadata: bool = True
    content_analysis: bool = True
    ml_analysis: bool = True
    ml_models: Optional[MLModelsConfig] = Field(default_factory=MLModelsConfig)
    quality: Optional[QualityConfig] = Field(default_factory=QualityConfig)


class DeduplicationConfig(BaseModel):
    """Deduplication configuration."""

    hash_algorithm: str = Field(default="sha256", pattern="^(sha256|md5)$")
    check_perceptual: bool = False
    keep_strategy: str = Field(
        default="highest_quality",
        pattern="^(highest_quality|oldest|newest)$"
    )
    safe_mode: bool = True


class RenamingConfig(BaseModel):
    """Renaming configuration."""

    pattern: str = "{date}_{time}"
    date_format: str = "%Y-%m-%d"
    time_format: str = "%H%M%S"
    datetime_format: str = "%Y-%m-%d_%H%M%S"
    collision_suffix: str = "_{seq:03d}"
    preserve_original: bool = True
    include_tags_in_filename: bool = False
    max_filename_length: int = Field(default=200, ge=50, le=255)


class FormattingConfig(BaseModel):
    """Format conversion configuration."""

    photo_format: str = Field(default="jpg", pattern="^(jpg|png)$")
    graphic_format: str = Field(default="png", pattern="^(jpg|png)$")
    jpeg_quality: int = Field(default=95, ge=0, le=100)
    png_compression: int = Field(default=6, ge=0, le=9)
    safe_conversion: bool = True
    supported_formats: List[str] = Field(
        default=[
            "jpg", "jpeg", "png", "heic", "heif", "webp",
            "bmp", "tiff", "raw", "cr2", "nef", "arw"
        ]
    )


class TaggingConfig(BaseModel):
    """Tag embedding configuration."""

    embed_tags: bool = True
    categories: Dict[str, bool] = Field(
        default={
            "scene": True,
            "objects": True,
            "quality": True,
            "location": True,
            "datetime": True,
        }
    )
    iptc_fields: Dict[str, bool] = Field(
        default={
            "keywords": True,
            "caption": True,
        }
    )
    min_confidence: float = Field(default=0.6, ge=0.0, le=1.0)


class ProcessingConfig(BaseModel):
    """Processing/threading configuration."""

    max_workers: int = Field(default=4, gt=0, le=32)
    batch_size: int = Field(default=100, gt=0)
    show_progress: bool = True
    verbose: bool = False


class VideoConfig(BaseModel):
    """Video analysis configuration."""

    keyframe_detection: bool = True
    keyframe_threshold: float = Field(default=30.0, ge=0.0)
    min_scene_duration: float = Field(default=1.0, ge=0.1)
    max_frames_per_video: int = Field(default=50, gt=0, le=1000)
    frames_per_scene: int = Field(default=3, gt=0, le=10)
    activity_inference: bool = True
    min_tag_percentage: float = Field(default=10.0, ge=0.0, le=100.0)
    supported_formats: List[str] = Field(
        default=["mp4", "avi", "mov", "mkv", "wmv", "webm", "m4v"]
    )


class NvidiaBuildConfig(BaseModel):
    """NVIDIA Build API integration configuration."""

    enabled: bool = False
    api_key: str = ""
    models: Dict[str, str] = Field(
        default={
            "retail_detection": "nvidia/retail-object-detection",
            "vision_language": "nvidia/nemotron-nano-12b-v2-vl",
        }
    )
    timeout: int = Field(default=30, gt=0, le=300)


class OutputConfig(BaseModel):
    """Output and logging configuration."""

    generate_report: bool = True
    report_format: str = Field(default="json", pattern="^(json|csv|html)$")
    log_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR)$")
    log_file: str = "image_engine.log"
    save_keyframes: bool = False
    keyframes_dir: str = "keyframes"


class AppConfig(BaseModel):
    """Complete application configuration."""

    model_config = ConfigDict(
        extra="allow",  # Allow additional fields for backwards compatibility
        validate_assignment=True,  # Validate on attribute assignment
    )

    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    deduplication: DeduplicationConfig = Field(default_factory=DeduplicationConfig)
    renaming: RenamingConfig = Field(default_factory=RenamingConfig)
    formatting: FormattingConfig = Field(default_factory=FormattingConfig)
    tagging: TaggingConfig = Field(default_factory=TaggingConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    video: Optional[VideoConfig] = Field(default_factory=VideoConfig)
    nvidia_build: Optional[NvidiaBuildConfig] = Field(default_factory=NvidiaBuildConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    def get_nested(self, key_path: str, default=None):
        """Get nested configuration value using dot notation."""
        keys = key_path.split('.')
        value = self.model_dump()

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value
