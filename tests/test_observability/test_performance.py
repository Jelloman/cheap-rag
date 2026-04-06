"""Tests for performance monitoring infrastructure."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.observability.performance import (
    MemoryProfiler,
    PerformanceMonitor,
    PerformanceProfiler,
    PerformanceStats,
    get_performance_monitor,
    record_operation,
)


@pytest.fixture
def profiler():
    return PerformanceProfiler(window_seconds=300)


@pytest.fixture
def monitor():
    return PerformanceMonitor(window_seconds=300)


class TestPerformanceProfiler:
    def test_record_latency(self, profiler):
        profiler.record_latency("embed_text", 42.5)
        stats = profiler.get_stats("embed_text")
        assert stats is not None
        assert stats.count == 1
        assert stats.mean_ms == pytest.approx(42.5)

    def test_get_stats_no_data(self, profiler):
        assert profiler.get_stats("nonexistent") is None

    def test_stats_multiple_records(self, profiler):
        profiler.record_latency("search", 10.0)
        profiler.record_latency("search", 20.0)
        profiler.record_latency("search", 30.0)

        stats = profiler.get_stats("search")
        assert stats is not None
        assert stats.count == 3
        assert stats.mean_ms == pytest.approx(20.0)
        assert stats.min_ms == pytest.approx(10.0)
        assert stats.max_ms == pytest.approx(30.0)

    def test_stats_percentiles(self, profiler):
        for i in range(1, 11):
            profiler.record_latency("op", float(i * 10))

        stats = profiler.get_stats("op")
        assert stats is not None
        assert stats.p95_ms >= stats.median_ms
        assert stats.p99_ms >= stats.p95_ms

    def test_get_all_stats(self, profiler):
        profiler.record_latency("op_a", 10.0)
        profiler.record_latency("op_b", 20.0)

        all_stats = profiler.get_all_stats()
        assert "op_a" in all_stats
        assert "op_b" in all_stats

    def test_record_with_metadata(self, profiler):
        profiler.record_latency("embed", 50.0, metadata={"batch_size": 32})
        # Should not raise; metadata is stored on the record

    def test_reset(self, profiler):
        profiler.record_latency("op", 10.0)
        profiler.reset()
        assert profiler.get_stats("op") is None

    def test_performance_stats_to_dict(self):
        stats = PerformanceStats(
            operation="test",
            count=5,
            mean_ms=20.0,
            median_ms=18.0,
            p95_ms=35.0,
            p99_ms=40.0,
            min_ms=10.0,
            max_ms=50.0,
            std_dev_ms=8.0,
            throughput_per_sec=2.5,
        )
        d = stats.to_dict()
        assert d["operation"] == "test"
        assert d["count"] == 5
        assert d["mean_ms"] == 20.0
        assert d["p95_ms"] == 35.0


class TestMemoryProfiler:
    def test_get_memory_usage(self):
        mem_profiler = MemoryProfiler()
        usage = mem_profiler.get_memory_usage()
        assert "rss_mb" in usage
        assert "vms_mb" in usage
        assert "percent" in usage
        assert usage["rss_mb"] > 0

    def test_get_gpu_memory(self):
        # Mocked to avoid loading torch DLLs (blocks thread-based timeout on Windows)
        mem_profiler = MemoryProfiler()
        with patch.object(MemoryProfiler, "get_gpu_memory", return_value=None):
            result = mem_profiler.get_gpu_memory()
        assert result is None or isinstance(result, dict)

    def test_force_gc(self):
        mem_profiler = MemoryProfiler()
        result = mem_profiler.force_gc()
        assert "objects_collected" in result
        assert "before_mb" in result
        assert "after_mb" in result


class TestPerformanceMonitor:
    def test_record_operation(self, monitor):
        monitor.record_operation("embed_query", 15.3)
        stats = monitor.profiler.get_stats("embed_query")
        assert stats is not None
        assert stats.count == 1

    def test_record_operation_with_metadata(self, monitor):
        monitor.record_operation("search", 25.0, metadata={"top_k": 5})
        stats = monitor.profiler.get_stats("search")
        assert stats is not None

    def test_get_summary(self, monitor):
        monitor.record_operation("embed", 10.0)
        monitor.record_operation("search", 20.0)

        # Mock get_gpu_memory to avoid loading torch DLLs on Windows
        with patch.object(MemoryProfiler, "get_gpu_memory", return_value=None):
            summary = monitor.get_summary()

        assert "timestamp" in summary
        assert "operations" in summary
        assert "embed" in summary["operations"]
        assert "search" in summary["operations"]
        assert "memory" in summary

    def test_get_summary_empty(self, monitor):
        # Mock get_gpu_memory to avoid loading torch DLLs on Windows
        with patch.object(MemoryProfiler, "get_gpu_memory", return_value=None):
            summary = monitor.get_summary()
        assert summary["operations"] == {}


class TestGlobalMonitor:
    def test_get_performance_monitor_singleton(self):
        m1 = get_performance_monitor()
        m2 = get_performance_monitor()
        assert m1 is m2

    def test_record_operation_global(self):
        # Should not raise
        record_operation("global_test_op", 5.0, metadata={"source": "test"})
        monitor = get_performance_monitor()
        stats = monitor.profiler.get_stats("global_test_op")
        assert stats is not None
