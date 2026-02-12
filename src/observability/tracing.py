"""OpenTelemetry tracing infrastructure for CHEAP RAG.

This module provides distributed tracing capabilities for the RAG pipeline,
enabling end-to-end visibility into embedding, retrieval, and generation stages.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any, ParamSpec, TypeVar

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)

# Type variables for generic decorators
P = ParamSpec("P")
R = TypeVar("R")


# Global tracer provider
_tracer_provider: TracerProvider | None = None
_initialized = False


def init_tracing(
    service_name: str = "cheap-rag",
    environment: str = "development",
    enable_console: bool = True,
    enable_otlp: bool = False,
    otlp_endpoint: str | None = None,
) -> None:
    """Initialize OpenTelemetry tracing.

    Args:
        service_name: Name of the service for trace identification
        environment: Environment name (development, production, etc.)
        enable_console: Whether to export traces to console
        enable_otlp: Whether to export traces to OTLP endpoint
        otlp_endpoint: OTLP endpoint URL (e.g., "http://localhost:4317")
    """
    global _tracer_provider, _initialized

    if _initialized:
        return

    # Create resource with service information
    resource = Resource.create(
        {
            "service.name": service_name,
            "service.version": "0.1.0",
            "deployment.environment": environment,
        }
    )

    # Create tracer provider
    _tracer_provider = TracerProvider(resource=resource)

    # Add console exporter for local development
    if enable_console:
        console_exporter = ConsoleSpanExporter()
        console_processor = SimpleSpanProcessor(console_exporter)
        _tracer_provider.add_span_processor(console_processor)

    # Add OTLP exporter for production
    if enable_otlp and otlp_endpoint:
        otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
        otlp_processor = BatchSpanProcessor(otlp_exporter)
        _tracer_provider.add_span_processor(otlp_processor)

    # Set as global tracer provider
    trace.set_tracer_provider(_tracer_provider)
    _initialized = True


def shutdown_tracing() -> None:
    """Shutdown tracing and flush remaining spans."""
    global _tracer_provider, _initialized

    if _tracer_provider:
        _tracer_provider.shutdown()
        _tracer_provider = None
        _initialized = False


def get_tracer(name: str = "cheap-rag") -> trace.Tracer:
    """Get a tracer instance.

    Args:
        name: Tracer name (typically module name)

    Returns:
        Tracer instance
    """
    if not _initialized:
        init_tracing()

    return trace.get_tracer(name)


@contextmanager
def trace_operation(
    operation_name: str,
    *,
    tracer: trace.Tracer | None = None,
    attributes: dict[str, Any] | None = None,
):
    """Context manager for tracing an operation.

    Example:
        with trace_operation("embed_query", attributes={"query_length": len(query)}):
            embedding = embed(query)

    Args:
        operation_name: Name of the operation being traced
        tracer: Optional tracer instance (creates new one if not provided)
        attributes: Optional attributes to add to the span

    Yields:
        The active span
    """
    if tracer is None:
        tracer = get_tracer()

    with tracer.start_as_current_span(operation_name) as span:
        # Add attributes if provided
        if attributes:
            for key, value in attributes.items():
                # Convert value to string if needed for span attributes
                if isinstance(value, (list, dict)):
                    span.set_attribute(key, str(value))
                else:
                    span.set_attribute(key, value)

        # Record start time
        start_time = time.perf_counter()

        try:
            yield span
        except Exception as e:
            # Record exception in span
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            raise
        finally:
            # Record duration
            duration_ms = (time.perf_counter() - start_time) * 1000
            span.set_attribute("duration_ms", duration_ms)


def trace_function(
    operation_name: str | None = None,
    *,
    tracer: trace.Tracer | None = None,
    include_args: bool = False,
    include_result: bool = False,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator for tracing a function.

    Example:
        @trace_function("embed_query", include_args=True)
        def embed_query(query: str) -> list[float]:
            ...

    Args:
        operation_name: Name for the operation (defaults to function name)
        tracer: Optional tracer instance
        include_args: Whether to include function arguments as span attributes
        include_result: Whether to include return value as span attribute

    Returns:
        Decorated function
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        nonlocal operation_name
        if operation_name is None:
            operation_name = func.__name__

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            attributes: dict[str, Any] = {}

            # Add function arguments if requested
            if include_args:
                attributes["function"] = func.__name__
                # Avoid logging large objects - just count them
                if args:
                    attributes["args_count"] = len(args)
                if kwargs:
                    attributes["kwargs_keys"] = list(kwargs.keys())

            with trace_operation(operation_name, tracer=tracer, attributes=attributes) as span:
                result = func(*args, **kwargs)

                # Add result info if requested (careful with large results)
                if include_result and result is not None:
                    if isinstance(result, (list, dict)):
                        span.set_attribute("result_type", type(result).__name__)
                        if isinstance(result, list):
                            span.set_attribute("result_length", len(result))
                    else:
                        span.set_attribute("result_type", type(result).__name__)

                return result

        return wrapper

    return decorator


def trace_async_function(
    operation_name: str | None = None,
    *,
    tracer: trace.Tracer | None = None,
    include_args: bool = False,
    include_result: bool = False,
):
    """Decorator for tracing an async function.

    Example:
        @trace_async_function("generate_answer")
        async def generate_answer(query: str) -> str:
            ...

    Args:
        operation_name: Name for the operation (defaults to function name)
        tracer: Optional tracer instance
        include_args: Whether to include function arguments as span attributes
        include_result: Whether to include return value as span attribute

    Returns:
        Decorated async function
    """

    def decorator(func):
        nonlocal operation_name
        if operation_name is None:
            operation_name = func.__name__

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            attributes: dict[str, Any] = {}

            if include_args:
                attributes["function"] = func.__name__
                if args:
                    attributes["args_count"] = len(args)
                if kwargs:
                    attributes["kwargs_keys"] = list(kwargs.keys())

            with trace_operation(operation_name, tracer=tracer, attributes=attributes) as span:
                result = await func(*args, **kwargs)

                if include_result and result is not None:
                    if isinstance(result, (list, dict)):
                        span.set_attribute("result_type", type(result).__name__)
                        if isinstance(result, list):
                            span.set_attribute("result_length", len(result))
                    else:
                        span.set_attribute("result_type", type(result).__name__)

                return result

        return wrapper

    return decorator


class TracingConfig:
    """Configuration for tracing behavior.

    Attributes:
        enabled: Whether tracing is enabled
        service_name: Service name for traces
        environment: Environment name
        console_export: Export to console
        otlp_export: Export to OTLP endpoint
        otlp_endpoint: OTLP endpoint URL
    """

    def __init__(
        self,
        enabled: bool = True,
        service_name: str = "cheap-rag",
        environment: str = "development",
        console_export: bool = True,
        otlp_export: bool = False,
        otlp_endpoint: str | None = None,
    ):
        self.enabled = enabled
        self.service_name = service_name
        self.environment = environment
        self.console_export = console_export
        self.otlp_export = otlp_export
        self.otlp_endpoint = otlp_endpoint

    def initialize(self) -> None:
        """Initialize tracing with this configuration."""
        if self.enabled:
            init_tracing(
                service_name=self.service_name,
                environment=self.environment,
                enable_console=self.console_export,
                enable_otlp=self.otlp_export,
                otlp_endpoint=self.otlp_endpoint,
            )
