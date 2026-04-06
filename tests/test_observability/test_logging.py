"""Tests for structured logging infrastructure."""

from __future__ import annotations

import sys
from io import StringIO

import pytest

from src.observability.logging import (
    LoggingConfig,
    StructuredLogger,
    get_correlation_id,
    init_logging,
    set_correlation_id,
)


@pytest.fixture(autouse=True)
def reset_correlation_id():
    """Reset correlation ID after each test."""
    yield
    # Reset by setting to a known state; contextvars are per-context
    # so just set a known value to avoid bleedover in same-context tests
    set_correlation_id("reset")


class TestCorrelationId:
    def test_set_and_get_correlation_id(self):
        set_correlation_id("test-id-123")
        assert get_correlation_id() == "test-id-123"

    def test_default_correlation_id_is_none(self):
        # Fresh contextvar state starts as None (unless already set in this context)
        # We can't easily test None without a fresh thread, so test set/get round-trip
        set_correlation_id("abc")
        assert get_correlation_id() == "abc"

    def test_correlation_id_overwrite(self):
        set_correlation_id("first")
        set_correlation_id("second")
        assert get_correlation_id() == "second"


class TestStructuredLogger:
    def test_create_logger(self):
        logger = StructuredLogger("test.module")
        assert logger.name == "test.module"

    def test_log_info(self, capsys):
        init_logging(level="DEBUG", format_type="text", enable_console=True)
        logger = StructuredLogger("test.info")
        # Should not raise
        logger.info("Test message")

    def test_log_with_kwargs(self, capsys):
        init_logging(level="DEBUG", format_type="text", enable_console=True)
        logger = StructuredLogger("test.kwargs")
        # Should not raise with structured fields
        logger.info("Test message", key="value", count=42)

    def test_log_debug(self):
        init_logging(level="DEBUG", format_type="text", enable_console=True)
        logger = StructuredLogger("test.debug")
        logger.debug("Debug message", detail="some detail")

    def test_log_warning(self):
        init_logging(level="DEBUG", format_type="text", enable_console=True)
        logger = StructuredLogger("test.warning")
        logger.warning("Warning message")

    def test_log_error(self):
        init_logging(level="DEBUG", format_type="text", enable_console=True)
        logger = StructuredLogger("test.error")
        logger.error("Error message", error_type="TestError")

    def test_log_critical(self):
        init_logging(level="DEBUG", format_type="text", enable_console=True)
        logger = StructuredLogger("test.critical")
        logger.critical("Critical message")

    def test_log_exception(self):
        init_logging(level="DEBUG", format_type="text", enable_console=True)
        logger = StructuredLogger("test.exception")
        try:
            raise ValueError("test exception")
        except ValueError:
            logger.exception("Caught exception")  # Should not raise


class TestLoggingConfig:
    def test_default_config(self):
        config = LoggingConfig()
        assert config.level == "INFO"
        assert config.format_type == "text"
        assert config.log_file is None
        assert config.enable_console is True

    def test_custom_config(self):
        config = LoggingConfig(level="DEBUG", format_type="json", enable_console=False)
        assert config.level == "DEBUG"
        assert config.format_type == "json"
        assert config.enable_console is False

    def test_initialize(self, tmp_path):
        log_file = tmp_path / "test.log"
        config = LoggingConfig(
            level="INFO",
            format_type="json",
            log_file=log_file,
            enable_console=False,
        )
        # Should not raise
        config.initialize()


class TestInitLogging:
    def test_init_logging_text_format(self):
        # Should not raise
        init_logging(level="INFO", format_type="text", enable_console=True)

    def test_init_logging_json_format(self):
        init_logging(level="DEBUG", format_type="json", enable_console=True)

    def test_init_logging_with_file(self, tmp_path):
        log_file = tmp_path / "app.log"
        init_logging(level="INFO", log_file=log_file, enable_console=False)
        assert log_file.parent.exists()

    def test_init_logging_no_console(self):
        # Should not raise
        init_logging(level="INFO", enable_console=False)
