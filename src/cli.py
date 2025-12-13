"""CLI entry point for backwards compatibility.

The actual CLI implementation is in src/ui/cli.py.
This module re-exports it to support:
- python -m src.cli
- nivo command (via setup.py entry point)
"""

from src.ui.cli import cli

__all__ = ["cli"]

if __name__ == "__main__":
    cli()
