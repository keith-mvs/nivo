"""Domain models for type-safe data structures."""

from .image_data import (
    ImageMetadata,
    ContentAnalysis,
    MLAnalysis,
    ImageAnalysisResult,
)
from .config_models import (
    AnalysisConfig,
    MLModelsConfig,
    DeduplicationConfig,
    RenamingConfig,
    TaggingConfig,
    ProcessingConfig,
    AppConfig,
)
from .processor_results import (
    RenameResult,
    TagEmbedResult,
    DuplicationResult,
    FormatConversionResult,
)

__all__ = [
    "ImageMetadata",
    "ContentAnalysis",
    "MLAnalysis",
    "ImageAnalysisResult",
    "AnalysisConfig",
    "MLModelsConfig",
    "DeduplicationConfig",
    "RenamingConfig",
    "TaggingConfig",
    "ProcessingConfig",
    "AppConfig",
    "RenameResult",
    "TagEmbedResult",
    "DuplicationResult",
    "FormatConversionResult",
]
