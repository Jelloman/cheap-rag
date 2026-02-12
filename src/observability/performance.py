"""Performance profiling and monitoring infrastructure.

This module provides detailed performance metrics including:
- Latency percentiles (p50, p95, p99)
- Throughput tracking (queries/sec)
- Memory usage monitoring
- GPU utilization tracking
- Per-stage profiling
"""

from __future__ import annotations

import gc
import os
import statistics
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.observability.logging import StructuredLogger


@dataclass
class LatencyRecord:
    """Record of a single operation's latency.

    Attributes:
        operation: Operation name
        latency_ms: Latency in milliseconds
        timestamp: When the operation occurred
        metadata: Additional context
    """

    operation: str
    latency_ms: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceStats:
    """Performance statistics for an operation.

    Attributes:
        operation: Operation name
        count: Number of observations
        mean_ms: Mean latency in milliseconds
        median_ms: Median latency (p50)
        p95_ms: 95th percentile latency
        p99_ms: 99th percentile latency
        min_ms: Minimum latency
        max_ms: Maximum latency
        std_dev_ms: Standard deviation
        throughput_per_sec: Operations per second
        metadata: Additional statistics
    """

    operation: str
    count: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    std_dev_ms: float
    throughput_per_sec: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "operation": self.operation,
            "count": self.count,
            "mean_ms": round(self.mean_ms, 2),
            "median_ms": round(self.median_ms, 2),
            "p95_ms": round(self.p95_ms, 2),
            "p99_ms": round(self.p99_ms, 2),
            "min_ms": round(self.min_ms, 2),
            "max_ms": round(self.max_ms, 2),
            "std_dev_ms": round(self.std_dev_ms, 2),
            "throughput_per_sec": round(self.throughput_per_sec, 2),
            "metadata": self.metadata,
        }


class PerformanceProfiler:
    """Track and analyze performance metrics.

    Thread-safe performance profiling with detailed statistics.
    """

    def __init__(self, window_seconds: int = 300):
        """Initialize performance profiler.

        Args:
            window_seconds: Time window for statistics (default: 5 minutes)
        """
        self.window_seconds = window_seconds
        self._records: defaultdict[str, list[LatencyRecord]] = defaultdict(list)
        self._lock = threading.Lock()
        self._logger = StructuredLogger("performance")

    def record_latency(
        self,
        operation: str,
        latency_ms: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record operation latency.

        Args:
            operation: Operation name
            latency_ms: Latency in milliseconds
            metadata: Additional context
        """
        record = LatencyRecord(
            operation=operation,
            latency_ms=latency_ms,
            timestamp=datetime.now(),
            metadata=metadata or {},
        )

        with self._lock:
            self._records[operation].append(record)

    def get_stats(self, operation: str) -> PerformanceStats | None:
        """Calculate statistics for an operation.

        Args:
            operation: Operation name

        Returns:
            PerformanceStats or None if no data
        """
        with self._lock:
            records = self._get_recent_records(operation)

            if not records:
                return None

            latencies = [r.latency_ms for r in records]

            # Calculate statistics
            count = len(latencies)
            mean_ms = statistics.mean(latencies)
            median_ms = statistics.median(latencies)
            min_ms = min(latencies)
            max_ms = max(latencies)

            # Standard deviation
            std_dev_ms = statistics.stdev(latencies) if count > 1 else 0.0

            # Percentiles
            sorted_latencies = sorted(latencies)
            p95_ms = sorted_latencies[int(0.95 * count)] if count > 0 else 0.0
            p99_ms = sorted_latencies[int(0.99 * count)] if count > 0 else 0.0

            # Throughput
            time_span = records[-1].timestamp.timestamp() - records[0].timestamp.timestamp()
            throughput_per_sec = count / time_span if time_span > 0 else 0.0

            return PerformanceStats(
                operation=operation,
                count=count,
                mean_ms=mean_ms,
                median_ms=median_ms,
                p95_ms=p95_ms,
                p99_ms=p99_ms,
                min_ms=min_ms,
                max_ms=max_ms,
                std_dev_ms=std_dev_ms,
                throughput_per_sec=throughput_per_sec,
            )

    def get_all_stats(self) -> dict[str, PerformanceStats]:
        """Get statistics for all tracked operations.

        Returns:
            Dictionary mapping operation name to stats
        """
        with self._lock:
            stats = {}
            for operation in self._records:
                operation_stats = self.get_stats(operation)
                if operation_stats:
                    stats[operation] = operation_stats
            return stats

    def clear_old_records(self) -> int:
        """Remove records outside the time window.

        Returns:
            Number of records removed
        """
        with self._lock:
            cutoff = datetime.now().timestamp() - self.window_seconds
            removed_count = 0

            for operation in list(self._records.keys()):
                original_count = len(self._records[operation])
                self._records[operation] = [
                    r for r in self._records[operation] if r.timestamp.timestamp() > cutoff
                ]
                removed_count += original_count - len(self._records[operation])

            return removed_count

    def reset(self) -> None:
        """Clear all performance records."""
        with self._lock:
            self._records.clear()

    def _get_recent_records(self, operation: str) -> list[LatencyRecord]:
        """Get records within the time window (internal, assumes lock held).

        Args:
            operation: Operation name

        Returns:
            List of latency records
        """
        cutoff = datetime.now().timestamp() - self.window_seconds
        return [r for r in self._records[operation] if r.timestamp.timestamp() > cutoff]


class MemoryProfiler:
    """Track memory usage statistics."""

    def __init__(self):
        """Initialize memory profiler."""
        self._logger = StructuredLogger("memory_profiler")

    def get_memory_usage(self) -> dict[str, float]:
        """Get current memory usage statistics.

        Returns:
            Dictionary with memory usage in MB
        """
        import psutil

        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()

        return {
            "rss_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size
            "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            "percent": process.memory_percent(),
        }

    def get_gpu_memory(self) -> dict[str, Any] | None:
        """Get GPU memory usage if available.

        Returns:
            Dictionary with GPU memory info or None if CUDA unavailable
        """
        try:
            import torch

            if not torch.cuda.is_available():
                return None

            return {
                "allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
                "reserved_mb": torch.cuda.memory_reserved() / 1024 / 1024,
                "max_allocated_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
            }
        except ImportError:
            return None

    def force_gc(self) -> dict[str, Any]:
        """Force garbage collection and return statistics.

        Returns:
            Dictionary with GC statistics
        """
        before = self.get_memory_usage()
        collected = gc.collect()
        after = self.get_memory_usage()

        return {
            "objects_collected": collected,
            "memory_freed_mb": before["rss_mb"] - after["rss_mb"],
            "before_mb": before["rss_mb"],
            "after_mb": after["rss_mb"],
        }


class PerformanceMonitor:
    """Unified performance monitoring combining profiler and memory tracking."""

    def __init__(self, window_seconds: int = 300):
        """Initialize performance monitor.

        Args:
            window_seconds: Time window for statistics
        """
        self.profiler = PerformanceProfiler(window_seconds)
        self.memory = MemoryProfiler()
        self._logger = StructuredLogger("performance_monitor")

    def record_operation(
        self,
        operation: str,
        duration_ms: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record an operation's performance.

        Args:
            operation: Operation name
            duration_ms: Duration in milliseconds
            metadata: Additional context
        """
        self.profiler.record_latency(operation, duration_ms, metadata)

    def get_summary(self) -> dict[str, Any]:
        """Get comprehensive performance summary.

        Returns:
            Dictionary with all performance metrics
        """
        stats = self.profiler.get_all_stats()
        memory = self.memory.get_memory_usage()
        gpu_memory = self.memory.get_gpu_memory()

        return {
            "timestamp": datetime.now().isoformat(),
            "operations": {op: stats[op].to_dict() for op in stats},
            "memory": memory,
            "gpu_memory": gpu_memory,
        }


# Global performance monitor
_global_monitor: PerformanceMonitor | None = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance.

    Returns:
        Global PerformanceMonitor instance
    """
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor


def record_operation(
    operation: str,
    duration_ms: float,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Record an operation to the global monitor.

    Args:
        operation: Operation name
        duration_ms: Duration in milliseconds
        metadata: Additional context
    """
    monitor = get_performance_monitor()
    monitor.record_operation(operation, duration_ms, metadata)
