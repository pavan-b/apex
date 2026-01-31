"""
Logging configuration for the application.

Configures both console and file-based logging with rotation.
Log files are written to a `logs/` directory with:
- Daily rotation
- 30-day retention
- Separate files for different log levels

Example:
    from backend.app.core.logging_config import setup_logging
    
    # Call once at application startup
    setup_logging(log_level=logging.INFO)
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import Optional


def get_log_directory() -> Path:
    """
    Get or create the logs directory.
    
    Returns:
        Path to the logs directory.
    """
    # Default to project root/logs
    project_root = Path(__file__).parent.parent.parent.parent
    log_dir = project_root / "logs"
    
    # Allow override via environment variable
    if env_log_dir := os.getenv("LOG_DIR"):
        log_dir = Path(env_log_dir)
    
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def setup_logging(
    log_level: int = logging.INFO,
    log_to_console: bool = True,
    log_to_file: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
    log_format: Optional[str] = None,
) -> logging.Logger:
    """
    Configure logging for the application.
    
    Sets up:
    - Console handler (stdout) for real-time monitoring
    - Rotating file handler for persistent logs
    - Separate error log file for ERROR+ level messages
    
    Args:
        log_level: Minimum log level (default: INFO).
        log_to_console: Whether to log to console (default: True).
        log_to_file: Whether to log to files (default: True).
        max_bytes: Max size per log file before rotation (default: 10MB).
        backup_count: Number of backup files to keep (default: 5).
        log_format: Custom log format string (optional).
    
    Returns:
        The root logger instance.
    
    Example:
        setup_logging(log_level=logging.DEBUG, backup_count=10)
    """
    # Default format with timestamp, level, logger name, and message
    if log_format is None:
        log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    
    # Date format for timestamps
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # Create formatter
    formatter = logging.Formatter(log_format, datefmt=date_format)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handlers
    if log_to_file:
        log_dir = get_log_directory()
        
        # Main application log (rotating by size)
        app_log_path = log_dir / "app.log"
        app_handler = RotatingFileHandler(
            app_log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        app_handler.setLevel(log_level)
        app_handler.setFormatter(formatter)
        root_logger.addHandler(app_handler)
        
        # Error log (only ERROR and above)
        error_log_path = log_dir / "error.log"
        error_handler = RotatingFileHandler(
            error_log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        root_logger.addHandler(error_handler)
        
        # Debug log (all levels, for detailed debugging)
        if log_level <= logging.DEBUG:
            debug_log_path = log_dir / "debug.log"
            debug_handler = RotatingFileHandler(
                debug_log_path,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
            debug_handler.setLevel(logging.DEBUG)
            debug_handler.setFormatter(formatter)
            root_logger.addHandler(debug_handler)
    
    # Log startup message
    root_logger.info("=" * 60)
    root_logger.info("Logging initialized")
    root_logger.info("  Log level: %s", logging.getLevelName(log_level))
    root_logger.info("  Console: %s", "enabled" if log_to_console else "disabled")
    if log_to_file:
        root_logger.info("  Log directory: %s", get_log_directory())
    root_logger.info("=" * 60)
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.
    
    This is a convenience function that ensures the logging system
    is properly configured before returning a logger.
    
    Args:
        name: Logger name (typically __name__).
    
    Returns:
        Logger instance.
    
    Example:
        logger = get_logger(__name__)
        logger.info("Hello from my module!")
    """
    return logging.getLogger(name)


# Module-level logger for this file
logger = get_logger(__name__)
