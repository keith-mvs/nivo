"""File processing workflow manager with backup and retention policies."""

import os
import shutil
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Callable
from dataclasses import dataclass, field
from enum import Enum


class RetentionPolicy(Enum):
    """Backup retention policies."""
    KEEP_FOREVER = "keep_forever"
    DAYS_7 = "days_7"
    DAYS_30 = "days_30"
    DAYS_90 = "days_90"
    AFTER_VERIFICATION = "after_verification"


@dataclass
class WorkflowConfig:
    """Configuration for file processing workflow."""
    start_directory: str  # Primary start directory (for backward compatibility)
    end_directory: str
    backup_directory: str
    retention_policy: RetentionPolicy = RetentionPolicy.DAYS_30
    file_patterns: List[str] = field(default_factory=lambda: ["*.jpg", "*.jpeg", "*.png", "*.heic", "*.heif"])
    create_manifest: bool = True
    verify_before_delete: bool = True
    # Multi-source support
    additional_directories: List[str] = field(default_factory=list)
    recursive: bool = False
    preserve_structure: bool = True  # Preserve folder structure in destination


@dataclass
class SourceFile:
    """Track a source file with its origin directory."""
    path: Path
    source_dir: Path
    relative_path: Path  # Path relative to source directory


@dataclass
class WorkflowResult:
    """Result of a workflow execution."""
    success: bool
    files_processed: int
    files_backed_up: int
    files_moved: int
    errors: List[str]
    manifest_path: Optional[str] = None
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    source_directories: List[str] = field(default_factory=list)


class FileProcessingWorkflow:
    """Manage file processing from start to end with backup and retention.

    Workflow stages:
    1. Scan source directory for matching files
    2. Create backup copies in backup directory
    3. Process files (apply transformations)
    4. Move processed files to end directory
    5. Create manifest for tracking
    6. Apply retention policy to old backups
    """

    def __init__(self, config: WorkflowConfig):
        """Initialize workflow with configuration.

        Args:
            config: WorkflowConfig with directory paths and settings
        """
        self.config = config
        self._processors: List[Callable] = []
        self._manifest: Dict = {}

    def add_processor(self, processor: Callable) -> "FileProcessingWorkflow":
        """Add a file processor to the pipeline.

        Args:
            processor: Callable that takes (source_path, dest_path) -> bool

        Returns:
            Self for chaining
        """
        self._processors.append(processor)
        return self

    def execute(self, dry_run: bool = False) -> WorkflowResult:
        """Execute the complete workflow.

        Args:
            dry_run: If True, preview actions without executing

        Returns:
            WorkflowResult with execution details
        """
        result = WorkflowResult(
            success=False,
            files_processed=0,
            files_backed_up=0,
            files_moved=0,
            errors=[],
        )

        try:
            # Validate directories
            self._validate_directories(dry_run)

            # Scan source files
            source_files = self._scan_source()
            if not source_files:
                result.success = True
                result.completed_at = datetime.now()
                return result

            # Create backup
            backed_up = self._backup_files(source_files, dry_run)
            result.files_backed_up = len(backed_up)

            # Process files
            processed = self._process_files(source_files, dry_run)
            result.files_processed = len(processed)

            # Move to destination
            moved = self._move_to_destination(processed, dry_run)
            result.files_moved = len(moved)

            # Create manifest
            if self.config.create_manifest and not dry_run:
                manifest_path = self._create_manifest(source_files, backed_up, moved)
                result.manifest_path = str(manifest_path)

            # Apply retention policy
            if not dry_run:
                self._apply_retention_policy()

            result.success = True

        except Exception as e:
            result.errors.append(str(e))

        result.completed_at = datetime.now()
        return result

    def _get_all_source_directories(self) -> List[Path]:
        """Get all source directories (primary + additional)."""
        dirs = [Path(self.config.start_directory)]
        for d in self.config.additional_directories:
            dirs.append(Path(d))
        return dirs

    def _validate_directories(self, dry_run: bool) -> None:
        """Validate and create required directories."""
        for source_dir in self._get_all_source_directories():
            if not source_dir.exists():
                raise ValueError(f"Source directory does not exist: {source_dir}")

        if not dry_run:
            Path(self.config.end_directory).mkdir(parents=True, exist_ok=True)
            Path(self.config.backup_directory).mkdir(parents=True, exist_ok=True)

    def _scan_source(self) -> List[SourceFile]:
        """Scan all source directories for matching files."""
        source_files = []

        for source_dir in self._get_all_source_directories():
            for pattern in self.config.file_patterns:
                # Recursive or flat scan
                glob_pattern = f"**/{pattern}" if self.config.recursive else pattern

                for filepath in source_dir.glob(glob_pattern):
                    if filepath.is_file():
                        relative = filepath.relative_to(source_dir)
                        source_files.append(SourceFile(
                            path=filepath,
                            source_dir=source_dir,
                            relative_path=relative,
                        ))

                # Also match uppercase extensions
                for filepath in source_dir.glob(glob_pattern.upper()):
                    if filepath.is_file():
                        relative = filepath.relative_to(source_dir)
                        source_files.append(SourceFile(
                            path=filepath,
                            source_dir=source_dir,
                            relative_path=relative,
                        ))

        # Deduplicate by path
        seen = set()
        unique = []
        for sf in source_files:
            if sf.path not in seen:
                seen.add(sf.path)
                unique.append(sf)

        return sorted(unique, key=lambda x: x.path)

    def _backup_files(self, files: List[SourceFile], dry_run: bool) -> List[Path]:
        """Create backup copies of source files."""
        backed_up = []
        backup_dir = Path(self.config.backup_directory)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_dir = backup_dir / f"backup_{timestamp}"

        if not dry_run:
            batch_dir.mkdir(parents=True, exist_ok=True)

        for source_file in files:
            if self.config.preserve_structure:
                # Preserve folder structure: source_dir_name/relative_path
                source_name = source_file.source_dir.name
                backup_subdir = batch_dir / source_name / source_file.relative_path.parent
                if not dry_run:
                    backup_subdir.mkdir(parents=True, exist_ok=True)
                backup_path = backup_subdir / source_file.path.name
            else:
                backup_path = batch_dir / source_file.path.name

            if not dry_run:
                shutil.copy2(source_file.path, backup_path)

            backed_up.append(backup_path)

        return backed_up

    def _process_files(self, files: List[SourceFile], dry_run: bool) -> List[SourceFile]:
        """Apply processors to files."""
        if not self._processors:
            return files  # No processing, pass through

        processed = []
        for source_file in files:
            success = True
            for processor in self._processors:
                if not dry_run:
                    try:
                        success = processor(source_file.path, source_file.path)
                    except Exception:
                        success = False
                        break

            if success:
                processed.append(source_file)

        return processed

    def _move_to_destination(self, files: List[SourceFile], dry_run: bool) -> List[Path]:
        """Move processed files to destination directory."""
        moved = []
        dest_dir = Path(self.config.end_directory)

        for source_file in files:
            if self.config.preserve_structure:
                # Preserve folder structure: source_dir_name/relative_path
                source_name = source_file.source_dir.name
                dest_subdir = dest_dir / source_name / source_file.relative_path.parent
                if not dry_run:
                    dest_subdir.mkdir(parents=True, exist_ok=True)
                dest_path = dest_subdir / source_file.path.name
            else:
                dest_path = dest_dir / source_file.path.name

            # Handle name collisions
            if dest_path.exists():
                stem = dest_path.stem
                suffix = dest_path.suffix
                counter = 1
                while dest_path.exists():
                    dest_path = dest_path.parent / f"{stem}_{counter:03d}{suffix}"
                    counter += 1

            if not dry_run:
                shutil.move(str(source_file.path), str(dest_path))

            moved.append(dest_path)

        return moved

    def _create_manifest(
        self,
        source_files: List[SourceFile],
        backed_up: List[Path],
        moved: List[Path],
    ) -> Path:
        """Create JSON manifest of the workflow execution."""
        manifest_dir = Path(self.config.backup_directory)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        manifest_path = manifest_dir / f"manifest_{timestamp}.json"

        # Group source files by directory
        sources_by_dir = {}
        for sf in source_files:
            dir_key = str(sf.source_dir)
            if dir_key not in sources_by_dir:
                sources_by_dir[dir_key] = []
            sources_by_dir[dir_key].append(str(sf.path))

        manifest = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "start_directory": self.config.start_directory,
                "additional_directories": self.config.additional_directories,
                "end_directory": self.config.end_directory,
                "backup_directory": self.config.backup_directory,
                "retention_policy": self.config.retention_policy.value,
                "recursive": self.config.recursive,
                "preserve_structure": self.config.preserve_structure,
            },
            "files": {
                "source_count": len(source_files),
                "backed_up_count": len(backed_up),
                "moved_count": len(moved),
                "sources_by_directory": sources_by_dir,
                "backup_files": [str(f) for f in backed_up],
                "destination_files": [str(f) for f in moved],
            },
        }

        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2)

        return manifest_path

    def _apply_retention_policy(self) -> int:
        """Apply retention policy to old backups.

        Returns:
            Number of backup directories deleted
        """
        policy = self.config.retention_policy
        if policy == RetentionPolicy.KEEP_FOREVER:
            return 0

        backup_dir = Path(self.config.backup_directory)
        deleted_count = 0

        # Calculate cutoff date
        cutoff_days = {
            RetentionPolicy.DAYS_7: 7,
            RetentionPolicy.DAYS_30: 30,
            RetentionPolicy.DAYS_90: 90,
        }.get(policy)

        if cutoff_days is None:
            return 0  # AFTER_VERIFICATION requires manual cleanup

        cutoff_date = datetime.now() - timedelta(days=cutoff_days)

        # Find and delete old backup directories
        for item in backup_dir.iterdir():
            if item.is_dir() and item.name.startswith("backup_"):
                try:
                    # Parse timestamp from directory name
                    ts_str = item.name.replace("backup_", "")
                    dir_date = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")

                    if dir_date < cutoff_date:
                        if self.config.verify_before_delete:
                            # Check for corresponding manifest verification
                            if not self._is_verified(item):
                                continue

                        shutil.rmtree(item)
                        deleted_count += 1

                except (ValueError, OSError):
                    continue  # Skip malformed directories

        return deleted_count

    def _is_verified(self, backup_dir: Path) -> bool:
        """Check if backup has been verified for deletion.

        Args:
            backup_dir: Path to backup directory

        Returns:
            True if verified, False otherwise
        """
        verification_file = backup_dir / ".verified"
        return verification_file.exists()

    @staticmethod
    def mark_verified(backup_dir: str) -> None:
        """Mark a backup directory as verified for deletion.

        Args:
            backup_dir: Path to backup directory to mark
        """
        verification_file = Path(backup_dir) / ".verified"
        verification_file.write_text(datetime.now().isoformat())

    def preview(self) -> Dict:
        """Preview workflow without executing.

        Returns:
            Dictionary describing planned actions
        """
        source_files = self._scan_source()

        # Group by source directory
        files_by_dir = {}
        for sf in source_files:
            dir_key = str(sf.source_dir)
            if dir_key not in files_by_dir:
                files_by_dir[dir_key] = 0
            files_by_dir[dir_key] += 1

        return {
            "source_directories": [self.config.start_directory] + self.config.additional_directories,
            "end_directory": self.config.end_directory,
            "backup_directory": self.config.backup_directory,
            "retention_policy": self.config.retention_policy.value,
            "recursive": self.config.recursive,
            "preserve_structure": self.config.preserve_structure,
            "files_to_process": len(source_files),
            "files_by_directory": files_by_dir,
            "file_patterns": self.config.file_patterns,
            "processors": len(self._processors),
            "sample_files": [str(sf.path) for sf in source_files[:10]],
        }


def create_workflow(
    start_directory: str,
    end_directory: str,
    backup_directory: str,
    retention_days: int = 30,
    additional_directories: Optional[List[str]] = None,
    recursive: bool = False,
    preserve_structure: bool = True,
) -> FileProcessingWorkflow:
    """Convenience function to create a workflow.

    Args:
        start_directory: Primary source directory path
        end_directory: Destination directory path
        backup_directory: Backup directory path
        retention_days: Days to keep backups (7, 30, 90, or 0 for forever)
        additional_directories: Additional source directories
        recursive: Scan directories recursively
        preserve_structure: Preserve folder structure in destination

    Returns:
        Configured FileProcessingWorkflow instance
    """
    policy_map = {
        0: RetentionPolicy.KEEP_FOREVER,
        7: RetentionPolicy.DAYS_7,
        30: RetentionPolicy.DAYS_30,
        90: RetentionPolicy.DAYS_90,
    }
    policy = policy_map.get(retention_days, RetentionPolicy.DAYS_30)

    config = WorkflowConfig(
        start_directory=start_directory,
        end_directory=end_directory,
        backup_directory=backup_directory,
        retention_policy=policy,
        additional_directories=additional_directories or [],
        recursive=recursive,
        preserve_structure=preserve_structure,
    )

    return FileProcessingWorkflow(config)


def create_library_workflow(
    source_directories: List[str],
    end_directory: str,
    backup_directory: str,
    retention_days: int = 30,
    recursive: bool = True,
    preserve_structure: bool = True,
) -> FileProcessingWorkflow:
    """Create workflow for image library with multiple source directories.

    Args:
        source_directories: List of source directory paths
        end_directory: Destination directory path
        backup_directory: Backup directory path
        retention_days: Days to keep backups (7, 30, 90, or 0 for forever)
        recursive: Scan directories recursively (default True for libraries)
        preserve_structure: Preserve folder structure in destination

    Returns:
        Configured FileProcessingWorkflow instance
    """
    if not source_directories:
        raise ValueError("At least one source directory is required")

    return create_workflow(
        start_directory=source_directories[0],
        end_directory=end_directory,
        backup_directory=backup_directory,
        retention_days=retention_days,
        additional_directories=source_directories[1:] if len(source_directories) > 1 else None,
        recursive=recursive,
        preserve_structure=preserve_structure,
    )
