"""Domain models for image processor results."""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path


@dataclass
class RenameResult:
    """Result from renaming operation."""

    # Original file info
    original_path: str
    original_name: str

    # New file info
    new_path: Optional[str] = None
    new_name: Optional[str] = None

    # Operation status
    success: bool = False
    renamed: bool = False
    collision: bool = False

    # Backup info
    backup_created: bool = False
    backup_path: Optional[str] = None

    # Error tracking
    error: Optional[str] = None

    # Metadata
    renamed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        if self.renamed_at:
            data['renamed_at'] = self.renamed_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RenameResult':
        """Create from dictionary."""
        if 'renamed_at' in data and isinstance(data['renamed_at'], str):
            data['renamed_at'] = datetime.fromisoformat(data['renamed_at'])
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


@dataclass
class TagEmbedResult:
    """Result from tag embedding operation."""

    # File info
    file_path: str

    # Tags embedded
    tags_embedded: List[str] = field(default_factory=list)
    tag_count: int = 0

    # IPTC fields updated
    iptc_keywords: bool = False
    iptc_caption: bool = False
    caption_text: Optional[str] = None

    # Operation status
    success: bool = False
    modified: bool = False

    # Backup info
    backup_created: bool = False
    backup_path: Optional[str] = None

    # Verification
    tags_verified: bool = False
    verification_error: Optional[str] = None

    # Error tracking
    error: Optional[str] = None

    # Metadata
    embedded_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        if self.embedded_at:
            data['embedded_at'] = self.embedded_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TagEmbedResult':
        """Create from dictionary."""
        if 'embedded_at' in data and isinstance(data['embedded_at'], str):
            data['embedded_at'] = datetime.fromisoformat(data['embedded_at'])
        if 'tags_embedded' not in data:
            data['tags_embedded'] = []
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


@dataclass
class DuplicationResult:
    """Result from deduplication operation."""

    # File info
    file_path: str
    file_hash: str

    # Duplicate info
    is_duplicate: bool = False
    duplicate_of: Optional[str] = None
    duplicates_found: List[str] = field(default_factory=list)
    duplicate_count: int = 0

    # Quality comparison (if keep_strategy=highest_quality)
    quality_score: Optional[float] = None
    kept_file: Optional[str] = None

    # Operation status
    deleted: bool = False
    kept: bool = False

    # Safe mode
    requires_confirmation: bool = True
    confirmed: bool = False

    # Error tracking
    error: Optional[str] = None

    # Metadata
    checked_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        if self.checked_at:
            data['checked_at'] = self.checked_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DuplicationResult':
        """Create from dictionary."""
        if 'checked_at' in data and isinstance(data['checked_at'], str):
            data['checked_at'] = datetime.fromisoformat(data['checked_at'])
        if 'duplicates_found' not in data:
            data['duplicates_found'] = []
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


@dataclass
class FormatConversionResult:
    """Result from format conversion operation."""

    # Original file info
    original_path: str
    original_format: str
    original_size: int

    # Converted file info
    converted_path: Optional[str] = None
    converted_format: Optional[str] = None
    converted_size: Optional[int] = None

    # Operation status
    success: bool = False
    converted: bool = False

    # Quality preservation
    quality_preserved: bool = False
    quality_loss: Optional[float] = None

    # Safe conversion
    original_kept: bool = False
    original_backup_path: Optional[str] = None

    # Performance metrics
    conversion_time: Optional[float] = None
    size_reduction: Optional[float] = None  # Percentage

    # Error tracking
    error: Optional[str] = None

    # Metadata
    converted_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        if self.converted_at:
            data['converted_at'] = self.converted_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FormatConversionResult':
        """Create from dictionary."""
        if 'converted_at' in data and isinstance(data['converted_at'], str):
            data['converted_at'] = datetime.fromisoformat(data['converted_at'])
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


@dataclass
class ProcessingBatchResult:
    """Aggregate results from batch processing."""

    # Batch info
    total_files: int
    files_processed: int
    files_failed: int

    # Individual results
    rename_results: List[RenameResult] = field(default_factory=list)
    tag_results: List[TagEmbedResult] = field(default_factory=list)
    dedup_results: List[DuplicationResult] = field(default_factory=list)
    conversion_results: List[FormatConversionResult] = field(default_factory=list)

    # Summary statistics
    files_renamed: int = 0
    files_tagged: int = 0
    duplicates_found: int = 0
    files_converted: int = 0

    # Performance metrics
    total_time: Optional[float] = None
    files_per_second: Optional[float] = None

    # Error tracking
    errors: List[str] = field(default_factory=list)

    # Metadata
    processed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total_files': self.total_files,
            'files_processed': self.files_processed,
            'files_failed': self.files_failed,
            'rename_results': [r.to_dict() for r in self.rename_results],
            'tag_results': [r.to_dict() for r in self.tag_results],
            'dedup_results': [r.to_dict() for r in self.dedup_results],
            'conversion_results': [r.to_dict() for r in self.conversion_results],
            'files_renamed': self.files_renamed,
            'files_tagged': self.files_tagged,
            'duplicates_found': self.duplicates_found,
            'files_converted': self.files_converted,
            'total_time': self.total_time,
            'files_per_second': self.files_per_second,
            'errors': self.errors,
            'processed_at': self.processed_at.isoformat() if self.processed_at else None,
        }
