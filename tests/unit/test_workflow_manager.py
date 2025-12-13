"""Tests for file processing workflow manager."""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta

from src.core.utils.workflow_manager import (
    FileProcessingWorkflow,
    WorkflowConfig,
    WorkflowResult,
    RetentionPolicy,
    create_workflow,
)


@pytest.fixture
def temp_dirs():
    """Create temporary directories for testing."""
    base = tempfile.mkdtemp()
    dirs = {
        "start": Path(base) / "start",
        "end": Path(base) / "end",
        "backup": Path(base) / "backup",
    }
    for d in dirs.values():
        d.mkdir(parents=True)

    yield dirs

    # Cleanup
    shutil.rmtree(base, ignore_errors=True)


@pytest.fixture
def sample_files(temp_dirs):
    """Create sample image files in start directory."""
    start_dir = temp_dirs["start"]
    files = []

    for i in range(5):
        filepath = start_dir / f"image_{i:03d}.jpg"
        filepath.write_bytes(b"fake image data")
        files.append(filepath)

    return files


class TestWorkflowConfig:
    """Tests for WorkflowConfig."""

    def test_config_creation(self, temp_dirs):
        """Create config with required fields."""
        config = WorkflowConfig(
            start_directory=str(temp_dirs["start"]),
            end_directory=str(temp_dirs["end"]),
            backup_directory=str(temp_dirs["backup"]),
        )

        assert config.retention_policy == RetentionPolicy.DAYS_30
        assert config.create_manifest is True

    def test_config_custom_retention(self, temp_dirs):
        """Create config with custom retention policy."""
        config = WorkflowConfig(
            start_directory=str(temp_dirs["start"]),
            end_directory=str(temp_dirs["end"]),
            backup_directory=str(temp_dirs["backup"]),
            retention_policy=RetentionPolicy.DAYS_7,
        )

        assert config.retention_policy == RetentionPolicy.DAYS_7


class TestFileProcessingWorkflow:
    """Tests for FileProcessingWorkflow."""

    def test_preview_empty_directory(self, temp_dirs):
        """Preview on empty directory."""
        config = WorkflowConfig(
            start_directory=str(temp_dirs["start"]),
            end_directory=str(temp_dirs["end"]),
            backup_directory=str(temp_dirs["backup"]),
        )
        workflow = FileProcessingWorkflow(config)
        preview = workflow.preview()

        assert preview["files_to_process"] == 0
        assert preview["processors"] == 0

    def test_preview_with_files(self, temp_dirs, sample_files):
        """Preview shows files to process."""
        config = WorkflowConfig(
            start_directory=str(temp_dirs["start"]),
            end_directory=str(temp_dirs["end"]),
            backup_directory=str(temp_dirs["backup"]),
        )
        workflow = FileProcessingWorkflow(config)
        preview = workflow.preview()

        assert preview["files_to_process"] == 5

    def test_execute_dry_run(self, temp_dirs, sample_files):
        """Dry run does not modify files."""
        config = WorkflowConfig(
            start_directory=str(temp_dirs["start"]),
            end_directory=str(temp_dirs["end"]),
            backup_directory=str(temp_dirs["backup"]),
        )
        workflow = FileProcessingWorkflow(config)
        result = workflow.execute(dry_run=True)

        assert result.success is True
        assert result.files_processed == 5
        # Files should still be in start directory
        assert len(list(temp_dirs["start"].glob("*.jpg"))) == 5
        # No files in end directory
        assert len(list(temp_dirs["end"].glob("*.jpg"))) == 0

    def test_execute_moves_files(self, temp_dirs, sample_files):
        """Execute moves files to destination."""
        config = WorkflowConfig(
            start_directory=str(temp_dirs["start"]),
            end_directory=str(temp_dirs["end"]),
            backup_directory=str(temp_dirs["backup"]),
            create_manifest=True,
            preserve_structure=False,  # Flat structure for simple test
        )
        workflow = FileProcessingWorkflow(config)
        result = workflow.execute(dry_run=False)

        assert result.success is True
        assert result.files_moved == 5
        # Files should be in end directory
        assert len(list(temp_dirs["end"].glob("*.jpg"))) == 5
        # Start directory should be empty
        assert len(list(temp_dirs["start"].glob("*.jpg"))) == 0

    def test_execute_creates_backup(self, temp_dirs, sample_files):
        """Execute creates backups."""
        config = WorkflowConfig(
            start_directory=str(temp_dirs["start"]),
            end_directory=str(temp_dirs["end"]),
            backup_directory=str(temp_dirs["backup"]),
            preserve_structure=False,  # Flat structure for simple test
        )
        workflow = FileProcessingWorkflow(config)
        result = workflow.execute(dry_run=False)

        assert result.files_backed_up == 5
        # Backup directory should have a batch folder
        backup_dirs = list(temp_dirs["backup"].glob("backup_*"))
        assert len(backup_dirs) == 1
        # Backup folder should have files
        assert len(list(backup_dirs[0].glob("*.jpg"))) == 5

    def test_execute_creates_manifest(self, temp_dirs, sample_files):
        """Execute creates manifest file."""
        config = WorkflowConfig(
            start_directory=str(temp_dirs["start"]),
            end_directory=str(temp_dirs["end"]),
            backup_directory=str(temp_dirs["backup"]),
            create_manifest=True,
        )
        workflow = FileProcessingWorkflow(config)
        result = workflow.execute(dry_run=False)

        assert result.manifest_path is not None
        assert Path(result.manifest_path).exists()

    def test_add_processor(self, temp_dirs, sample_files):
        """Add custom processor to workflow."""
        config = WorkflowConfig(
            start_directory=str(temp_dirs["start"]),
            end_directory=str(temp_dirs["end"]),
            backup_directory=str(temp_dirs["backup"]),
        )
        workflow = FileProcessingWorkflow(config)

        processed_files = []

        def tracker(src, dst):
            processed_files.append(src)
            return True

        workflow.add_processor(tracker)
        result = workflow.execute(dry_run=False)

        assert len(processed_files) == 5
        assert result.files_processed == 5

    def test_invalid_start_directory(self, temp_dirs):
        """Invalid start directory raises error."""
        config = WorkflowConfig(
            start_directory="/nonexistent/path",
            end_directory=str(temp_dirs["end"]),
            backup_directory=str(temp_dirs["backup"]),
        )
        workflow = FileProcessingWorkflow(config)
        result = workflow.execute(dry_run=False)

        assert result.success is False
        assert len(result.errors) > 0


class TestRetentionPolicy:
    """Tests for backup retention policies."""

    def test_keep_forever_no_deletion(self, temp_dirs, sample_files):
        """KEEP_FOREVER policy does not delete backups."""
        config = WorkflowConfig(
            start_directory=str(temp_dirs["start"]),
            end_directory=str(temp_dirs["end"]),
            backup_directory=str(temp_dirs["backup"]),
            retention_policy=RetentionPolicy.KEEP_FOREVER,
        )
        workflow = FileProcessingWorkflow(config)
        workflow.execute(dry_run=False)

        # Create an old backup directory manually
        old_backup = temp_dirs["backup"] / "backup_20200101_000000"
        old_backup.mkdir()
        (old_backup / "test.jpg").write_bytes(b"old backup")

        # Run retention
        deleted = workflow._apply_retention_policy()

        assert deleted == 0
        assert old_backup.exists()

    def test_mark_verified(self, temp_dirs):
        """Mark backup as verified for deletion."""
        backup_dir = temp_dirs["backup"] / "backup_20200101_000000"
        backup_dir.mkdir()

        FileProcessingWorkflow.mark_verified(str(backup_dir))

        verification_file = backup_dir / ".verified"
        assert verification_file.exists()


class TestCreateWorkflowConvenience:
    """Tests for create_workflow convenience function."""

    def test_create_workflow_default(self, temp_dirs):
        """Create workflow with defaults."""
        workflow = create_workflow(
            start_directory=str(temp_dirs["start"]),
            end_directory=str(temp_dirs["end"]),
            backup_directory=str(temp_dirs["backup"]),
        )

        assert workflow.config.retention_policy == RetentionPolicy.DAYS_30

    def test_create_workflow_custom_retention(self, temp_dirs):
        """Create workflow with custom retention days."""
        workflow = create_workflow(
            start_directory=str(temp_dirs["start"]),
            end_directory=str(temp_dirs["end"]),
            backup_directory=str(temp_dirs["backup"]),
            retention_days=7,
        )

        assert workflow.config.retention_policy == RetentionPolicy.DAYS_7

    def test_create_workflow_keep_forever(self, temp_dirs):
        """Create workflow with keep forever policy."""
        workflow = create_workflow(
            start_directory=str(temp_dirs["start"]),
            end_directory=str(temp_dirs["end"]),
            backup_directory=str(temp_dirs["backup"]),
            retention_days=0,
        )

        assert workflow.config.retention_policy == RetentionPolicy.KEEP_FOREVER


class TestWorkflowResult:
    """Tests for WorkflowResult."""

    def test_result_attributes(self):
        """WorkflowResult has expected attributes."""
        result = WorkflowResult(
            success=True,
            files_processed=10,
            files_backed_up=10,
            files_moved=10,
            errors=[],
        )

        assert result.success is True
        assert result.files_processed == 10
        assert result.started_at is not None
        assert result.completed_at is None

    def test_result_with_errors(self):
        """WorkflowResult captures errors."""
        result = WorkflowResult(
            success=False,
            files_processed=5,
            files_backed_up=5,
            files_moved=0,
            errors=["Permission denied", "File not found"],
        )

        assert result.success is False
        assert len(result.errors) == 2


class TestMultiDirectoryWorkflow:
    """Tests for multiple source directory support."""

    @pytest.fixture
    def multi_source_dirs(self):
        """Create multiple source directories for testing."""
        base = tempfile.mkdtemp()
        dirs = {
            "source1": Path(base) / "source1",
            "source2": Path(base) / "source2",
            "end": Path(base) / "end",
            "backup": Path(base) / "backup",
        }
        for d in dirs.values():
            d.mkdir(parents=True)

        # Create files in source1
        for i in range(3):
            (dirs["source1"] / f"photo_{i}.jpg").write_bytes(b"jpeg data")

        # Create files in source2
        for i in range(2):
            (dirs["source2"] / f"image_{i}.png").write_bytes(b"png data")

        yield dirs

        shutil.rmtree(base, ignore_errors=True)

    def test_multiple_directories_preview(self, multi_source_dirs):
        """Preview shows files from multiple directories."""
        from src.core.utils.workflow_manager import create_library_workflow

        workflow = create_library_workflow(
            source_directories=[
                str(multi_source_dirs["source1"]),
                str(multi_source_dirs["source2"]),
            ],
            end_directory=str(multi_source_dirs["end"]),
            backup_directory=str(multi_source_dirs["backup"]),
            recursive=False,
        )
        preview = workflow.preview()

        assert preview["files_to_process"] == 5
        assert len(preview["source_directories"]) == 2

    def test_multiple_directories_execute(self, multi_source_dirs):
        """Execute processes files from multiple directories."""
        from src.core.utils.workflow_manager import create_library_workflow

        workflow = create_library_workflow(
            source_directories=[
                str(multi_source_dirs["source1"]),
                str(multi_source_dirs["source2"]),
            ],
            end_directory=str(multi_source_dirs["end"]),
            backup_directory=str(multi_source_dirs["backup"]),
            recursive=False,
        )
        result = workflow.execute(dry_run=False)

        assert result.success is True
        assert result.files_moved == 5

    def test_preserve_structure(self, multi_source_dirs):
        """Files preserve folder structure from source."""
        from src.core.utils.workflow_manager import create_library_workflow

        workflow = create_library_workflow(
            source_directories=[
                str(multi_source_dirs["source1"]),
                str(multi_source_dirs["source2"]),
            ],
            end_directory=str(multi_source_dirs["end"]),
            backup_directory=str(multi_source_dirs["backup"]),
            preserve_structure=True,
        )
        workflow.execute(dry_run=False)

        # Files should be in subdirectories named after source dirs
        end_dir = multi_source_dirs["end"]
        assert (end_dir / "source1").exists()
        assert (end_dir / "source2").exists()

    def test_recursive_scanning(self):
        """Recursive scanning finds files in subdirectories."""
        base = tempfile.mkdtemp()
        try:
            source = Path(base) / "source"
            subdir = source / "2024" / "vacation"
            subdir.mkdir(parents=True)

            (source / "root.jpg").write_bytes(b"root file")
            (subdir / "beach.jpg").write_bytes(b"beach photo")

            end_dir = Path(base) / "end"
            backup_dir = Path(base) / "backup"

            workflow = create_workflow(
                start_directory=str(source),
                end_directory=str(end_dir),
                backup_directory=str(backup_dir),
                recursive=True,
                preserve_structure=False,
            )
            preview = workflow.preview()

            assert preview["files_to_process"] == 2
            assert preview["recursive"] is True

        finally:
            shutil.rmtree(base, ignore_errors=True)


class TestLibraryWorkflow:
    """Tests for create_library_workflow convenience function."""

    def test_library_workflow_requires_directories(self):
        """Library workflow requires at least one source."""
        from src.core.utils.workflow_manager import create_library_workflow

        with pytest.raises(ValueError, match="At least one source"):
            create_library_workflow(
                source_directories=[],
                end_directory="/tmp/end",
                backup_directory="/tmp/backup",
            )

    def test_library_workflow_defaults_recursive(self, temp_dirs):
        """Library workflow defaults to recursive scanning."""
        from src.core.utils.workflow_manager import create_library_workflow

        workflow = create_library_workflow(
            source_directories=[str(temp_dirs["start"])],
            end_directory=str(temp_dirs["end"]),
            backup_directory=str(temp_dirs["backup"]),
        )

        assert workflow.config.recursive is True
        assert workflow.config.preserve_structure is True
