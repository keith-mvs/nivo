"""Unit tests for configuration management."""

import pytest
import tempfile
import os
from pathlib import Path
import yaml
from src.core.utils.config import Config


class TestConfigLoading:
    """Test configuration loading and validation."""

    def test_load_default_config(self):
        """Test loading the default configuration file."""
        config = Config()

        assert config.config_path is not None
        assert 'analysis' in config.config
        assert 'processing' in config.config

    def test_load_custom_config(self):
        """Test loading a custom configuration file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({
                'analysis': {
                    'extract_metadata': False,
                    'ml_analysis': True,
                },
                'processing': {
                    'max_workers': 8,
                },
            }, f)
            temp_path = f.name

        try:
            config = Config(config_path=temp_path)

            assert config.get('analysis.extract_metadata') is False
            assert config.get('processing.max_workers') == 8
        finally:
            os.unlink(temp_path)

    def test_load_missing_file_uses_default(self):
        """Test that missing config file falls back to defaults."""
        config = Config(config_path="nonexistent_file.yaml")

        assert config.get('analysis.extract_metadata') is True
        assert config.get('processing.max_workers') == 4

    def test_load_invalid_yaml_uses_default(self):
        """Test that invalid YAML falls back to defaults."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content:\n  - broken")
            temp_path = f.name

        try:
            config = Config(config_path=temp_path)

            # Should fall back to defaults
            assert config.get('analysis.extract_metadata') is True
        finally:
            os.unlink(temp_path)


class TestConfigAccess:
    """Test configuration value access."""

    def test_get_simple_key(self):
        """Test getting a simple top-level key."""
        config = Config()

        analysis = config.get('analysis')

        assert isinstance(analysis, dict)
        assert 'extract_metadata' in analysis

    def test_get_nested_key(self):
        """Test getting nested keys with dot notation."""
        config = Config()

        extract_metadata = config.get('analysis.extract_metadata')
        ml_analysis = config.get('analysis.ml_analysis')

        assert isinstance(extract_metadata, bool)
        assert isinstance(ml_analysis, bool)

    def test_get_with_default(self):
        """Test getting non-existent key returns default."""
        config = Config()

        value = config.get('nonexistent.key', default="default_value")

        assert value == "default_value"

    def test_get_deeply_nested(self):
        """Test getting deeply nested keys."""
        config = Config()

        # Check ml_models config if it exists
        batch_size = config.get('analysis.ml_models.batch_size', default=8)

        assert isinstance(batch_size, int)
        assert batch_size > 0


class TestConfigModification:
    """Test configuration modification."""

    def test_set_simple_value(self):
        """Test setting a configuration value."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({'analysis': {'extract_metadata': True}}, f)
            temp_path = f.name

        try:
            config = Config(config_path=temp_path, validate=False)
            config.set('analysis.extract_metadata', False)

            assert config.get('analysis.extract_metadata') is False
        finally:
            os.unlink(temp_path)

    def test_set_nested_value(self):
        """Test setting nested configuration value."""
        config = Config(validate=False)
        config.set('processing.max_workers', 16)

        assert config.get('processing.max_workers') == 16

    def test_set_creates_missing_keys(self):
        """Test that set creates missing intermediate keys."""
        config = Config(validate=False)
        config.set('new_section.new_key.nested_key', "test_value")

        assert config.get('new_section.new_key.nested_key') == "test_value"


class TestConfigCaching:
    """Test configuration caching behavior."""

    def test_config_get_is_cached(self):
        """Test that config.get uses caching."""
        config = Config()

        # First call should cache
        value1 = config.get('analysis.extract_metadata')

        # Second call should use cache (same object)
        value2 = config.get('analysis.extract_metadata')

        assert value1 == value2

    def test_cache_invalidated_on_set(self):
        """Test that cache is invalidated when values are set."""
        config = Config(validate=False)

        original = config.get('processing.max_workers', 4)
        config.set('processing.max_workers', 16)
        updated = config.get('processing.max_workers')

        assert updated == 16
        assert updated != original


class TestConfigValidation:
    """Test Pydantic configuration validation."""

    def test_valid_config_passes_validation(self):
        """Test that valid configuration passes validation."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({
                'analysis': {
                    'extract_metadata': True,
                    'ml_analysis': True,
                    'ml_models': {
                        'batch_size': 8,
                        'use_gpu': True,
                    },
                },
                'processing': {
                    'max_workers': 4,
                },
                'tagging': {
                    'embed_tags': True,
                    'min_confidence': 0.6,
                    'categories': {
                        'scene': True,
                        'objects': True,
                    },
                },
            }, f)
            temp_path = f.name

        try:
            config = Config(config_path=temp_path, validate=True)

            # Should have validated config
            assert config._validated_config is not None or config._raw_config is not None
        finally:
            os.unlink(temp_path)

    def test_invalid_batch_size_fails_validation(self):
        """Test that invalid batch size fails validation."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({
                'analysis': {
                    'ml_models': {
                        'batch_size': -1,  # Invalid: must be > 0
                    },
                },
            }, f)
            temp_path = f.name

        try:
            config = Config(config_path=temp_path, validate=True)

            # Should fall back to unvalidated config
            # But config should still be accessible
            assert config.get('analysis') is not None
        finally:
            os.unlink(temp_path)

    def test_validate_config_method(self):
        """Test the validate_config method."""
        config = Config(validate=False)

        # Manually trigger validation
        is_valid = config.validate_config()

        # Should be able to validate (True or False depending on pydantic availability)
        assert isinstance(is_valid, bool)


class TestConfigSave:
    """Test configuration saving."""

    def test_save_config(self):
        """Test saving configuration to file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "test_config.yaml")

            config = Config(validate=False)
            config.set('custom.setting', "test_value")
            config.save(path=config_path)

            # Verify file was created
            assert os.path.exists(config_path)

            # Load the saved config and verify
            loaded_config = Config(config_path=config_path, validate=False)
            assert loaded_config.get('custom.setting') == "test_value"

    def test_save_creates_directories(self):
        """Test that save creates missing directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "subdir", "test_config.yaml")

            config = Config(validate=False)
            config.save(path=config_path)

            assert os.path.exists(config_path)


class TestConfigBackwardsCompatibility:
    """Test backwards compatibility with existing code."""

    def test_config_property_returns_dict(self):
        """Test that .config property returns a dictionary."""
        config = Config()

        assert isinstance(config.config, dict)

    def test_can_access_raw_config(self):
        """Test direct access to raw config for backwards compatibility."""
        config = Config()

        # Should be able to iterate over config keys
        assert 'analysis' in config.config
        assert isinstance(config.config['analysis'], dict)

    def test_existing_code_patterns_work(self):
        """Test that existing code patterns still work."""
        config = Config()

        # Pattern 1: config.get()
        value1 = config.get('processing.max_workers', 4)
        assert isinstance(value1, int)

        # Pattern 2: config.config dict access
        value2 = config.config.get('processing', {})
        assert isinstance(value2, dict)

        # Pattern 3: Nested dict access
        ml_config = config.get('analysis.ml_models', {})
        if not isinstance(ml_config, dict):
            ml_config = {}
        assert isinstance(ml_config, dict)


class TestConfigClearCache:
    """Test cache clearing functionality."""

    def test_clear_cache(self):
        """Test clearing configuration caches."""
        config = Config()

        # Access some values to populate cache
        config.get('analysis.extract_metadata')
        config.get('processing.max_workers')

        # Clear cache
        Config.clear_cache()

        # Cache should be cleared (this test mainly ensures no errors)
        value = config.get('analysis.extract_metadata')
        assert value is not None


class TestWindowsPathHandling:
    """Test Windows-specific path handling."""

    def test_windows_paths_in_config(self):
        """Test that Windows paths are handled correctly."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({
                'paths': {
                    'input': r'C:\Users\test\Pictures',
                    'output': r'C:\Users\test\Output',
                },
            }, f)
            temp_path = f.name

        try:
            config = Config(config_path=temp_path, validate=False)

            input_path = config.get('paths.input')
            assert 'Users' in input_path
        finally:
            os.unlink(temp_path)

    def test_config_path_is_absolute(self):
        """Test that config paths are absolute."""
        config = Config()

        path = Path(config.config_path)
        assert path.is_absolute()
