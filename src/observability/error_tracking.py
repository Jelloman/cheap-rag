"""Error tracking and monitoring infrastructure.

This module provides comprehensive error tracking with:
- Error rate calculation by component and error type
- Error aggregation and reporting
- Integration with logging and tracing
"""

from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.observability.logging import StructuredLogger


@dataclass
class ErrorRecord:
    """Record of a single error occurrence.

    Attributes:
        timestamp: When the error occurred
        component: Component where error occurred (e.g., "embeddings", "retrieval")
        error_type: Type of error (class name)
        error_message: Error message
        correlation_id: Optional correlation ID for request tracking
        metadata: Additional error context
    """

    timestamp: datetime
    component: str
    error_type: str
    error_message: str
    correlation_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "component": self.component,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "correlation_id": self.correlation_id,
            "metadata": self.metadata,
        }


class ErrorTracker:
    """Track and aggregate errors across the system.

    Thread-safe error tracking with rate calculation and reporting.
    """

    def __init__(self, window_seconds: int = 300):
        """Initialize error tracker.

        Args:
            window_seconds: Time window for rate calculation (default: 5 minutes)
        """
        self.window_seconds = window_seconds
        self._errors: list[ErrorRecord] = []
        self._lock = threading.Lock()
        self._logger = StructuredLogger("error_tracker")

    def record_error(
        self,
        component: str,
        error: Exception,
        correlation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record an error occurrence.

        Args:
            component: Component where error occurred
            error: Exception instance
            correlation_id: Optional correlation ID
            metadata: Additional error context
        """
        record = ErrorRecord(
            timestamp=datetime.now(),
            component=component,
            error_type=type(error).__name__,
            error_message=str(error),
            correlation_id=correlation_id,
            metadata=metadata or {},
        )

        with self._lock:
            self._errors.append(record)

        # Log the error
        self._logger.error(
            f"Error in {component}",
            error_type=record.error_type,
            error_message=record.error_message,
            correlation_id=correlation_id,
        )

    def get_error_rate(self, component: str | None = None) -> float:
        """Calculate error rate (errors per minute) in the time window.

        Args:
            component: Optional component filter

        Returns:
            Error rate (errors per minute)
        """
        with self._lock:
            recent_errors = self._get_recent_errors()

            if component:
                recent_errors = [e for e in recent_errors if e.component == component]

            if not recent_errors:
                return 0.0

            return len(recent_errors) / (self.window_seconds / 60)

    def get_error_counts_by_type(self, component: str | None = None) -> dict[str, int]:
        """Get error counts grouped by error type.

        Args:
            component: Optional component filter

        Returns:
            Dictionary mapping error type to count
        """
        with self._lock:
            recent_errors = self._get_recent_errors()

            if component:
                recent_errors = [e for e in recent_errors if e.component == component]

            counts: dict[str, int] = defaultdict(int)
            for error in recent_errors:
                counts[error.error_type] += 1

            return dict(counts)

    def get_error_counts_by_component(self) -> dict[str, int]:
        """Get error counts grouped by component.

        Returns:
            Dictionary mapping component to count
        """
        with self._lock:
            recent_errors = self._get_recent_errors()

            counts: dict[str, int] = defaultdict(int)
            for error in recent_errors:
                counts[error.component] += 1

            return dict(counts)

    def get_recent_errors(
        self,
        component: str | None = None,
        limit: int | None = None,
    ) -> list[ErrorRecord]:
        """Get recent errors within the time window.

        Args:
            component: Optional component filter
            limit: Optional limit on number of errors returned

        Returns:
            List of error records (most recent first)
        """
        with self._lock:
            recent_errors = self._get_recent_errors()

            if component:
                recent_errors = [e for e in recent_errors if e.component == component]

            # Sort by timestamp descending (most recent first)
            recent_errors.sort(key=lambda e: e.timestamp, reverse=True)

            if limit:
                recent_errors = recent_errors[:limit]

            return recent_errors

    def clear_old_errors(self) -> int:
        """Remove errors outside the time window.

        Returns:
            Number of errors removed
        """
        with self._lock:
            cutoff = datetime.now().timestamp() - self.window_seconds
            original_count = len(self._errors)

            self._errors = [e for e in self._errors if e.timestamp.timestamp() > cutoff]

            removed = original_count - len(self._errors)
            return removed

    def reset(self) -> None:
        """Clear all error records."""
        with self._lock:
            self._errors.clear()

    def _get_recent_errors(self) -> list[ErrorRecord]:
        """Get errors within the time window (internal, assumes lock held).

        Returns:
            List of error records
        """
        cutoff = datetime.now().timestamp() - self.window_seconds
        return [e for e in self._errors if e.timestamp.timestamp() > cutoff]

    def get_summary(self) -> dict[str, Any]:
        """Get error summary statistics.

        Returns:
            Dictionary with error statistics
        """
        with self._lock:
            recent_errors = self._get_recent_errors()

            return {
                "total_errors": len(recent_errors),
                "error_rate_per_minute": len(recent_errors) / (self.window_seconds / 60),
                "errors_by_component": self.get_error_counts_by_component(),
                "errors_by_type": self.get_error_counts_by_type(),
                "window_seconds": self.window_seconds,
                "timestamp": datetime.now().isoformat(),
            }


# Global error tracker instance
_global_tracker: ErrorTracker | None = None


def get_error_tracker() -> ErrorTracker:
    """Get the global error tracker instance.

    Returns:
        Global ErrorTracker instance
    """
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = ErrorTracker()
    return _global_tracker


def record_error(
    component: str,
    error: Exception,
    correlation_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Record an error to the global tracker.

    Args:
        component: Component where error occurred
        error: Exception instance
        correlation_id: Optional correlation ID
        metadata: Additional error context
    """
    tracker = get_error_tracker()
    tracker.record_error(component, error, correlation_id, metadata)
