"""Domain models for image analysis results."""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path


@dataclass
class ImageMetadata:
    """Metadata extracted from image EXIF/file system."""

    # File information
    file_name: str
    file_path: str
    file_size: int

    # Image dimensions
    width: int
    height: int
    format: Optional[str] = None
    mode: Optional[str] = None
    megapixels: Optional[float] = None

    # Date/time
    datetime_original: Optional[datetime] = None
    datetime_modified: Optional[datetime] = None

    # GPS
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude: Optional[float] = None
    gps_coordinates: Optional[str] = None

    # Camera information
    make: Optional[str] = None
    model: Optional[str] = None
    lens_model: Optional[str] = None
    f_number: Optional[float] = None
    exposure_time: Optional[Any] = None  # Can be tuple or float
    iso_speed_ratings: Optional[int] = None
    focal_length: Optional[Any] = None  # Can be tuple or float

    # Additional EXIF
    orientation: Optional[int] = None
    flash: Optional[Any] = None
    white_balance: Optional[Any] = None

    # Error tracking
    error: Optional[str] = None
    exif_error: Optional[str] = None
    gps_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, handling datetime serialization."""
        data = asdict(self)
        # Convert datetime objects to ISO format strings
        if self.datetime_original:
            data['datetime_original'] = self.datetime_original.isoformat()
        if self.datetime_modified:
            data['datetime_modified'] = self.datetime_modified.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ImageMetadata':
        """Create from dictionary, handling datetime parsing."""
        # Parse datetime strings back to datetime objects
        if 'datetime_original' in data and isinstance(data['datetime_original'], str):
            data['datetime_original'] = datetime.fromisoformat(data['datetime_original'])
        if 'datetime_modified' in data and isinstance(data['datetime_modified'], str):
            data['datetime_modified'] = datetime.fromisoformat(data['datetime_modified'])

        # Extract only fields that ImageMetadata expects
        valid_fields = {k: v for k, v in data.items() if k in cls.__annotations__}
        return cls(**valid_fields)


@dataclass
class ContentAnalysis:
    """Content-based image analysis results."""

    # Perceptual hashes for similarity detection
    phash: Optional[str] = None
    average_hash: Optional[str] = None
    dhash: Optional[str] = None
    whash: Optional[str] = None

    # Sharpness/blur detection
    sharpness_score: Optional[float] = None
    is_blurry: Optional[bool] = None
    sharpness_level: Optional[str] = None  # very_blurry, slightly_blurry, acceptable, sharp, very_sharp

    # Quality metrics
    quality_score: Optional[float] = None
    noise_level: Optional[float] = None
    dynamic_range: Optional[int] = None
    brightness: Optional[float] = None
    contrast: Optional[float] = None
    exposure: Optional[str] = None  # underexposed, well_exposed, overexposed, etc.

    # Color analysis
    dominant_colors: Optional[List[Dict[str, Any]]] = None  # List of {rgb, percentage}
    average_color: Optional[List[int]] = None  # [R, G, B]
    color_temperature: Optional[str] = None  # warm, cool, neutral
    warmth_score: Optional[float] = None

    # Error tracking
    error: Optional[str] = None
    hash_error: Optional[str] = None
    blur_error: Optional[str] = None
    quality_error: Optional[str] = None
    color_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContentAnalysis':
        """Create from dictionary."""
        valid_fields = {k: v for k, v in data.items() if k in cls.__annotations__}
        return cls(**valid_fields)


@dataclass
class MLAnalysis:
    """Machine learning vision analysis results."""

    # Scene classification
    primary_scene: Optional[str] = None
    scene_scores: Dict[str, float] = field(default_factory=dict)

    # Object detection
    objects: List[Dict[str, Any]] = field(default_factory=list)  # List of {label, confidence, bbox}
    object_counts: Dict[str, int] = field(default_factory=dict)  # {label: count}

    # Generated tags
    tags: List[str] = field(default_factory=list)

    # Model information
    scene_model: Optional[str] = None
    object_model: Optional[str] = None

    # Performance metrics
    inference_time: Optional[float] = None
    gpu_memory_used: Optional[float] = None

    # Error tracking
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MLAnalysis':
        """Create from dictionary."""
        # Ensure default factories are applied for missing fields
        if 'scene_scores' not in data:
            data['scene_scores'] = {}
        if 'objects' not in data:
            data['objects'] = []
        if 'object_counts' not in data:
            data['object_counts'] = {}
        if 'tags' not in data:
            data['tags'] = []

        valid_fields = {k: v for k, v in data.items() if k in cls.__annotations__}
        return cls(**valid_fields)


@dataclass
class ImageAnalysisResult:
    """Complete analysis result combining all phases."""

    # Required image path
    image_path: str

    # Analysis phase results
    metadata: Optional[ImageMetadata] = None
    content: Optional[ContentAnalysis] = None
    ml_vision: Optional[MLAnalysis] = None

    # Overall status
    analysis_complete: bool = False
    phases_completed: List[str] = field(default_factory=list)

    # Error tracking
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with nested conversions."""
        return {
            'image_path': self.image_path,
            'metadata': self.metadata.to_dict() if self.metadata else None,
            'content': self.content.to_dict() if self.content else None,
            'ml_vision': self.ml_vision.to_dict() if self.ml_vision else None,
            'analysis_complete': self.analysis_complete,
            'phases_completed': self.phases_completed,
            'errors': self.errors,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ImageAnalysisResult':
        """Create from dictionary with nested parsing."""
        return cls(
            image_path=data['image_path'],
            metadata=ImageMetadata.from_dict(data['metadata']) if data.get('metadata') else None,
            content=ContentAnalysis.from_dict(data['content']) if data.get('content') else None,
            ml_vision=MLAnalysis.from_dict(data['ml_vision']) if data.get('ml_vision') else None,
            analysis_complete=data.get('analysis_complete', False),
            phases_completed=data.get('phases_completed', []),
            errors=data.get('errors', []),
        )

    def add_error(self, error: str):
        """Add an error message."""
        self.errors.append(error)

    def mark_phase_complete(self, phase: str):
        """Mark a phase as completed."""
        if phase not in self.phases_completed:
            self.phases_completed.append(phase)

        # Check if all phases are complete
        expected_phases = ['metadata', 'content', 'ml_vision']
        if all(p in self.phases_completed for p in expected_phases):
            self.analysis_complete = True
