"""Image Engine - Intelligent photo management system."""

__version__ = "1.0.0"

# Re-export main components for backward compatibility
from .core.engine import ImageEngine
from .ui.cli import cli

__all__ = ["ImageEngine", "cli", "__version__"]
