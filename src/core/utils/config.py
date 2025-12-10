"""Configuration management for Image Engine."""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional


class Config:
    """Configuration manager for Image Engine."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.

        Args:
            config_path: Path to custom config file. Uses default if None.
        """
        self.config_path = config_path or self._get_default_config_path()
        self.config = self._load_config()

    def _get_default_config_path(self) -> str:
        """Get path to default configuration file."""
        # Navigate from src/core/utils/ to project root
        project_root = Path(__file__).parent.parent.parent.parent
        return str(project_root / "config" / "default_config.yaml")

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config or {}
        except FileNotFoundError:
            print(f"Warning: Config file not found at {self.config_path}")
            return self._get_default_config()
        except yaml.YAMLError as e:
            print(f"Error parsing config file: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Return minimal default configuration."""
        return {
            "analysis": {
                "extract_metadata": True,
                "content_analysis": True,
                "ml_analysis": True,
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
            },
            "processing": {
                "max_workers": 4,
                "show_progress": True,
            },
        }

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key_path: Configuration key path (e.g., 'analysis.ml_analysis')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key_path.split('.')
        value = self.config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def set(self, key_path: str, value: Any):
        """
        Set configuration value using dot notation.

        Args:
            key_path: Configuration key path (e.g., 'analysis.ml_analysis')
            value: Value to set
        """
        keys = key_path.split('.')
        config = self.config

        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        config[keys[-1]] = value

    def save(self, path: Optional[str] = None):
        """
        Save configuration to file.

        Args:
            path: Path to save config. Uses current config_path if None.
        """
        save_path = path or self.config_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
