"""Observability module for tracing, metrics, and logging."""

from __future__ import annotations

from src.observability.error_tracking import (
    ErrorRecord,
    ErrorTracker,
    get_error_tracker,
    record_error,
)
from src.observability.logging import (
    LoggingConfig,
    StructuredLogger,
    get_correlation_id,
    init_logging,
    set_correlation_id,
)
from src.observability.performance import (
    MemoryProfiler,
    PerformanceMonitor,
    PerformanceProfiler,
    PerformanceStats,
    get_performance_monitor,
    record_operation,
)
from src.observability.tracing import (
    TracingConfig,
    get_tracer,
    init_tracing,
    shutdown_tracing,
    trace_async_function,
    trace_function,
    trace_operation,
)

__all__ = [
    # Tracing
    "init_tracing",
    "shutdown_tracing",
    "get_tracer",
    "trace_operation",
    "trace_function",
    "trace_async_function",
    "TracingConfig",
    # Logging
    "init_logging",
    "StructuredLogger",
    "set_correlation_id",
    "get_correlation_id",
    "LoggingConfig",
    # Error Tracking
    "ErrorTracker",
    "ErrorRecord",
    "get_error_tracker",
    "record_error",
    # Performance
    "PerformanceMonitor",
    "PerformanceProfiler",
    "PerformanceStats",
    "MemoryProfiler",
    "get_performance_monitor",
    "record_operation",
]
