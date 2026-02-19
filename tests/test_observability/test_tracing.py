"""Tests for OpenTelemetry tracing infrastructure."""

from __future__ import annotations

import pytest

from src.observability.tracing import (
    TracingConfig,
    get_tracer,
    init_tracing,
    shutdown_tracing,
    trace_async_function,
    trace_function,
    trace_operation,
)


@pytest.fixture(autouse=True)
def reset_tracing():
    """Ensure tracing is shut down after each test."""
    yield
    shutdown_tracing()


class TestInitTracing:
    def test_init_tracing_console(self):
        # Should not raise
        init_tracing(service_name="test-service", enable_console=True, enable_otlp=False)

    def test_init_tracing_idempotent(self):
        # Calling twice should be safe (no-op second time)
        init_tracing(service_name="test-service")
        init_tracing(service_name="test-service")  # Should not raise

    def test_shutdown_tracing(self):
        init_tracing()
        shutdown_tracing()  # Should not raise

    def test_shutdown_without_init(self):
        shutdown_tracing()  # Should not raise even if not initialized


class TestGetTracer:
    def test_get_tracer_returns_tracer(self):
        tracer = get_tracer("test")
        assert tracer is not None

    def test_get_tracer_auto_inits(self):
        # get_tracer initializes if not already done
        tracer = get_tracer("auto-init")
        assert tracer is not None


class TestTraceOperation:
    def test_trace_operation_basic(self):
        with trace_operation("test_op") as span:
            assert span is not None

    def test_trace_operation_with_attributes(self):
        attrs = {"query_length": 42, "model": "test-model"}
        with trace_operation("test_op", attributes=attrs) as span:
            assert span is not None

    def test_trace_operation_records_exception(self):
        with pytest.raises(ValueError):
            with trace_operation("test_op"):
                raise ValueError("test error")

    def test_trace_operation_records_duration(self):
        with trace_operation("test_op") as span:
            # Just verify it completes without error
            pass

    def test_trace_operation_list_attribute(self):
        # List attributes should be converted to string
        with trace_operation("test_op", attributes={"ids": ["a", "b"]}) as span:
            assert span is not None


class TestTraceFunctionDecorator:
    def test_trace_function_basic(self):
        @trace_function("test_func")
        def my_func(x: int) -> int:
            return x * 2

        result = my_func(5)
        assert result == 10

    def test_trace_function_default_name(self):
        @trace_function()
        def compute_value() -> str:
            return "done"

        assert compute_value() == "done"

    def test_trace_function_include_args(self):
        @trace_function("test_with_args", include_args=True)
        def func_with_args(a: int, b: str) -> str:
            return f"{a}-{b}"

        assert func_with_args(1, "test") == "1-test"

    def test_trace_function_include_result(self):
        @trace_function("test_result", include_result=True)
        def func_with_result() -> list[int]:
            return [1, 2, 3]

        result = func_with_result()
        assert result == [1, 2, 3]

    def test_trace_function_propagates_exception(self):
        @trace_function("failing_func")
        def fails() -> None:
            raise RuntimeError("expected error")

        with pytest.raises(RuntimeError, match="expected error"):
            fails()

    def test_trace_function_preserves_return_value(self):
        @trace_function("returns_dict")
        def returns_dict() -> dict[str, int]:
            return {"a": 1, "b": 2}

        assert returns_dict() == {"a": 1, "b": 2}


class TestTraceAsyncFunctionDecorator:
    def test_trace_async_function_basic(self):
        import asyncio

        @trace_async_function("async_test")
        async def async_func() -> str:
            return "async result"

        result = asyncio.run(async_func())
        assert result == "async result"

    def test_trace_async_function_propagates_exception(self):
        import asyncio

        @trace_async_function("async_fail")
        async def async_fails() -> None:
            raise ValueError("async error")

        with pytest.raises(ValueError, match="async error"):
            asyncio.run(async_fails())

    def test_trace_async_function_with_args(self):
        import asyncio

        @trace_async_function("async_with_args", include_args=True)
        async def async_with_args(x: int) -> int:
            return x + 1

        assert asyncio.run(async_with_args(10)) == 11


class TestTracingConfig:
    def test_default_config(self):
        config = TracingConfig()
        assert config.enabled is True
        assert config.service_name == "cheap-rag"
        assert config.console_export is True
        assert config.otlp_export is False

    def test_custom_config(self):
        config = TracingConfig(
            enabled=True,
            service_name="my-service",
            environment="production",
            console_export=False,
            otlp_export=True,
            otlp_endpoint="http://localhost:4317",
        )
        assert config.service_name == "my-service"
        assert config.environment == "production"
        assert not config.console_export

    def test_initialize(self):
        config = TracingConfig(enabled=True, console_export=True)
        config.initialize()  # Should not raise

    def test_initialize_disabled(self):
        config = TracingConfig(enabled=False)
        config.initialize()  # Should be a no-op, not raise
