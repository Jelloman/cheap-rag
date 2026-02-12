"""Structured logging infrastructure for CHEAP RAG.

This module provides structured logging with context propagation,
correlation IDs, and integration with OpenTelemetry traces.
"""

from __future__ import annotations

import contextvars
import sys
from pathlib import Path
from typing import Any

from loguru import logger

# Context variable for request/correlation ID
correlation_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "correlation_id", default=None
)


def set_correlation_id(correlation_id: str) -> None:
    """Set correlation ID for current context.

    Args:
        correlation_id: Correlation ID (e.g., request ID, trace ID)
    """
    correlation_id_var.set(correlation_id)


def get_correlation_id() -> str | None:
    """Get correlation ID for current context.

    Returns:
        Correlation ID or None if not set
    """
    return correlation_id_var.get()


def correlation_id_filter(record: dict[str, Any]) -> bool:
    """Add correlation ID to log record.

    Args:
        record: Log record dictionary

    Returns:
        Always True to allow the record through
    """
    record["extra"]["correlation_id"] = get_correlation_id() or "N/A"
    return True


def init_logging(
    level: str = "INFO",
    format_type: str = "text",
    log_file: str | Path | None = None,
    enable_console: bool = True,
) -> None:
    """Initialize structured logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type: Format type ("text" or "json")
        log_file: Optional log file path
        enable_console: Whether to log to console
    """
    # Remove default handler
    logger.remove()

    # Define formats
    if format_type == "json":
        log_format = (
            '{"time": "{time:YYYY-MM-DD HH:mm:ss.SSS}", '
            '"level": "{level}", '
            '"correlation_id": "{extra[correlation_id]}", '
            '"name": "{name}", '
            '"function": "{function}", '
            '"line": {line}, '
            '"message": "{message}"}\n'
        )
    else:
        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{extra[correlation_id]}</cyan> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>\n"
        )

    # Add console handler
    if enable_console:
        logger.add(
            sys.stderr,
            format=log_format,
            level=level,
            filter=correlation_id_filter,
            colorize=(format_type == "text"),
        )

    # Add file handler if specified
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # Always use JSON for file output for easier parsing
        json_format = (
            '{"time": "{time:YYYY-MM-DD HH:mm:ss.SSS}", '
            '"level": "{level}", '
            '"correlation_id": "{extra[correlation_id]}", '
            '"name": "{name}", '
            '"function": "{function}", '
            '"line": {line}, '
            '"message": "{message}"}\n'
        )

        logger.add(
            log_file,
            format=json_format,
            level=level,
            filter=correlation_id_filter,
            rotation="100 MB",
            retention="30 days",
            compression="zip",
        )


class StructuredLogger:
    """Wrapper for structured logging with additional context.

    Example:
        logger = StructuredLogger("cheap_rag.embeddings")
        logger.info("Embedding query", query_length=len(query), model=model_name)
    """

    def __init__(self, name: str):
        """Initialize logger with name.

        Args:
            name: Logger name (typically module name)
        """
        self.name = name
        self._logger = logger.bind(name=name)

    def _log(self, level: str, message: str, **kwargs: Any) -> None:
        """Internal log method with structured fields.

        Args:
            level: Log level
            message: Log message
            **kwargs: Additional structured fields
        """
        extra = {"name": self.name}
        extra.update(kwargs)

        # Get the appropriate log method
        log_method = getattr(self._logger, level.lower())

        # Log with structured fields
        if kwargs:
            fields_str = " | " + " | ".join(f"{k}={v}" for k, v in kwargs.items())
            log_method(message + fields_str)
        else:
            log_method(message)

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        self._log("DEBUG", message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        self._log("INFO", message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        self._log("WARNING", message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message."""
        self._log("ERROR", message, **kwargs)

    def critical(self, message: str, **kwargs: Any) -> None:
        """Log critical message."""
        self._log("CRITICAL", message, **kwargs)

    def exception(self, message: str, **kwargs: Any) -> None:
        """Log exception with traceback."""
        self._logger.exception(message, **kwargs)


class LoggingConfig:
    """Configuration for logging behavior.

    Attributes:
        level: Log level
        format_type: Format type (text or json)
        log_file: Optional log file path
        enable_console: Enable console logging
    """

    def __init__(
        self,
        level: str = "INFO",
        format_type: str = "text",
        log_file: str | Path | None = None,
        enable_console: bool = True,
    ):
        self.level = level
        self.format_type = format_type
        self.log_file = log_file
        self.enable_console = enable_console

    def initialize(self) -> None:
        """Initialize logging with this configuration."""
        init_logging(
            level=self.level,
            format_type=self.format_type,
            log_file=self.log_file,
            enable_console=self.enable_console,
        )
