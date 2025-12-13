"""Configuration management for Image Engine with Pydantic validation."""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from functools import lru_cache
from pydantic import ValidationError

# Import Pydantic models
try:
    from ..models.config_models import AppConfig
except ImportError:
    # Fallback if models not available yet
    AppConfig = None


class Config:
    """Configuration manager with validation and caching."""

    # Class-level cache for loaded configs
    _config_cache: Dict[str, 'Config'] = {}

    def __init__(self, config_path: Optional[str] = None, validate: bool = True):
        """
        Initialize configuration.

        Args:
            config_path: Path to custom config file. Uses default if None.
            validate: Enable Pydantic validation (recommended)
        """
        self.config_path = config_path or self._get_default_config_path()
        self.validate = validate
        self._raw_config = self._load_config()
        self._validated_config: Optional[AppConfig] = None

        # Attempt validation if enabled and models available
        if self.validate and AppConfig is not None:
            try:
                self._validated_config = AppConfig(**self._raw_config)
                print(f"Configuration validated successfully: {self.config_path}")
            except ValidationError as e:
                print(f"ERROR: Configuration validation failed:")
                print(e)
                print("\nFalling back to unvalidated configuration.")
                self._validated_config = None

        # Cache this instance
        Config._config_cache[self.config_path] = self

    @staticmethod
    def _get_default_config_path() -> str:
        """Get path to default configuration file."""
        # Navigate from src/core/utils/ to project root
        project_root = Path(__file__).parent.parent.parent.parent
        return str(project_root / "config" / "default_config.yaml")

    @lru_cache(maxsize=1)
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file with caching."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            if not config:
                print(f"Warning: Empty config file at {self.config_path}")
                return self._get_default_config()

            return config

        except FileNotFoundError:
            print(f"Warning: Config file not found at {self.config_path}")
            return self._get_default_config()

        except yaml.YAMLError as e:
            print(f"Error parsing config file: {e}")
            print("Using default configuration.")
            return self._get_default_config()

        except Exception as e:
            print(f"Unexpected error loading config: {e}")
            return self._get_default_config()

    @staticmethod
    def _get_default_config() -> Dict[str, Any]:
        """Return minimal default configuration."""
        return {
            "analysis": {
                "extract_metadata": True,
                "content_analysis": True,
                "ml_analysis": True,
                "ml_models": {
                    "use_gpu": True,
                    "batch_size": 8,
                },
            },
            "deduplication": {
                "hash_algorithm": "sha256",
                "safe_mode": True,
            },
            "renaming": {
                "pattern": "{datetime}",
                "datetime_format": "%Y-%m-%d_%H%M%S",
            },
            "formatting": {
                "photo_format": "jpg",
                "jpeg_quality": 95,
            },
            "tagging": {
                "embed_tags": True,
                "min_confidence": 0.6,
                "categories": {
                    "scene": True,
                    "objects": True,
                },
            },
            "processing": {
                "max_workers": 4,
                "show_progress": True,
            },
        }

    @property
    def config(self) -> Dict[str, Any]:
        """Get raw config dictionary for backwards compatibility."""
        if self._validated_config:
            return self._validated_config.model_dump()
        return self._raw_config

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation with caching.

        Args:
            key_path: Configuration key path (e.g., 'analysis.ml_analysis')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        # Use validated config if available
        if self._validated_config:
            return self._validated_config.get_nested(key_path, default)

        # Fallback to raw config
        return self._get_from_dict(key_path, default)

    # Sentinel for cache-compatible None default
    _NOT_FOUND = object()

    @lru_cache(maxsize=128)
    def _get_from_dict_cached(self, key_path: str) -> Any:
        """Cached helper for dictionary access (returns _NOT_FOUND if missing)."""
        keys = key_path.split('.')
        value = self._raw_config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return Config._NOT_FOUND

        return value

    def _get_from_dict(self, key_path: str, default: Any = None) -> Any:
        """Get value from dict with unhashable default support."""
        result = self._get_from_dict_cached(key_path)
        if result is Config._NOT_FOUND:
            return default
        return result

    def set(self, key_path: str, value: Any):
        """
        Set configuration value using dot notation.

        Args:
            key_path: Configuration key path (e.g., 'analysis.ml_analysis')
            value: Value to set
        """
        keys = key_path.split('.')
        config = self._raw_config

        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        config[keys[-1]] = value

        # Invalidate cache on set
        self._get_from_dict_cached.cache_clear()

        # Re-validate if validation is enabled
        if self.validate and AppConfig is not None:
            try:
                self._validated_config = AppConfig(**self._raw_config)
            except ValidationError as e:
                print(f"Warning: Configuration no longer valid after set: {e}")
                self._validated_config = None

    def save(self, path: Optional[str] = None):
        """
        Save configuration to file.

        Args:
            path: Path to save config. Uses current config_path if None.
        """
        save_path = path or self.config_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(self._raw_config, f, default_flow_style=False, indent=2)

    def validate_config(self) -> bool:
        """
        Validate configuration against Pydantic models.

        Returns:
            True if valid, False otherwise
        """
        if AppConfig is None:
            print("Pydantic models not available for validation")
            return False

        try:
            AppConfig(**self._raw_config)
            return True
        except ValidationError as e:
            print(f"Configuration validation failed:")
            print(e)
            return False

    @classmethod
    def clear_cache(cls):
        """Clear all configuration caches."""
        cls._config_cache.clear()
        # Clear lru_cache for all instances
        for instance in cls._config_cache.values():
            instance._get_from_dict.cache_clear()
            instance._load_config.cache_clear()
