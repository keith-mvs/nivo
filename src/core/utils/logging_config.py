"""Centralized logging configuration for Nivo Image Engine."""

import logging
import sys
from pathlib import Path
from typing import Optional


# Module-level logger cache
_loggers: dict[str, logging.Logger] = {}

# Default format for console output
CONSOLE_FORMAT = "%(message)s"
VERBOSE_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEBUG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Get or create a logger with the given name.

    Args:
        name: Logger name, typically __name__ from the calling module

    Returns:
        Configured logger instance
    """
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    _loggers[name] = logger
    return logger


def configure_logging(
    level: int = logging.INFO,
    verbose: bool = False,
    debug: bool = False,
    log_file: Optional[Path] = None,
    quiet: bool = False
) -> None:
    """Configure global logging settings.

    Args:
        level: Base logging level
        verbose: Use verbose format with timestamps
        debug: Use debug format with file/line info
        log_file: Optional file path for log output
        quiet: Suppress all output except errors
    """
    # Determine format
    if debug:
        fmt = DEBUG_FORMAT
        level = logging.DEBUG
    elif verbose:
        fmt = VERBOSE_FORMAT
    else:
        fmt = CONSOLE_FORMAT

    if quiet:
        level = logging.ERROR

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(fmt))
    root_logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        file_handler.setFormatter(logging.Formatter(DEBUG_FORMAT))
        root_logger.addHandler(file_handler)

    # Suppress noisy third-party loggers
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("ultralytics").setLevel(logging.WARNING)


def set_level(level: int) -> None:
    """Set logging level for all Nivo loggers.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    for handler in root_logger.handlers:
        handler.setLevel(level)


# Initialize with default configuration on import
configure_logging()
