"""Tests for error tracking infrastructure."""

from __future__ import annotations

import time

import pytest

from src.observability.error_tracking import (
    ErrorRecord,
    ErrorTracker,
    get_error_tracker,
    record_error,
)


@pytest.fixture
def tracker():
    """Create a fresh ErrorTracker for each test."""
    return ErrorTracker(window_seconds=60)


class TestErrorRecord:
    def test_to_dict(self):
        from datetime import datetime

        record = ErrorRecord(
            timestamp=datetime(2026, 1, 1, 12, 0, 0),
            component="embeddings",
            error_type="ValueError",
            error_message="bad value",
            correlation_id="abc-123",
            metadata={"query": "test"},
        )
        d = record.to_dict()
        assert d["component"] == "embeddings"
        assert d["error_type"] == "ValueError"
        assert d["error_message"] == "bad value"
        assert d["correlation_id"] == "abc-123"
        assert d["metadata"] == {"query": "test"}
        assert "timestamp" in d


class TestErrorTracker:
    def test_record_error(self, tracker):
        tracker.record_error("embeddings", ValueError("test error"))
        assert len(tracker._errors) == 1

    def test_record_error_with_correlation_id(self, tracker):
        tracker.record_error("retrieval", RuntimeError("oops"), correlation_id="req-1")
        errors = tracker.get_recent_errors()
        assert errors[0].correlation_id == "req-1"

    def test_record_error_with_metadata(self, tracker):
        tracker.record_error(
            "generation",
            ConnectionError("timeout"),
            metadata={"model": "qwen"},
        )
        errors = tracker.get_recent_errors()
        assert errors[0].metadata == {"model": "qwen"}

    def test_get_recent_errors_empty(self, tracker):
        assert tracker.get_recent_errors() == []

    def test_get_recent_errors_with_component_filter(self, tracker):
        tracker.record_error("embeddings", ValueError("e1"))
        tracker.record_error("retrieval", ValueError("e2"))
        tracker.record_error("embeddings", ValueError("e3"))

        emb_errors = tracker.get_recent_errors(component="embeddings")
        assert len(emb_errors) == 2
        assert all(e.component == "embeddings" for e in emb_errors)

    def test_get_recent_errors_with_limit(self, tracker):
        for i in range(5):
            tracker.record_error("api", ValueError(f"error {i}"))

        limited = tracker.get_recent_errors(limit=3)
        assert len(limited) == 3

    def test_get_recent_errors_sorted_newest_first(self, tracker):
        tracker.record_error("api", ValueError("first"))
        time.sleep(0.01)
        tracker.record_error("api", ValueError("second"))

        errors = tracker.get_recent_errors()
        assert errors[0].error_message == "second"
        assert errors[1].error_message == "first"

    def test_error_rate_no_errors(self, tracker):
        assert tracker.get_error_rate() == 0.0

    def test_error_rate_with_errors(self, tracker):
        tracker.record_error("api", ValueError("e1"))
        tracker.record_error("api", ValueError("e2"))
        rate = tracker.get_error_rate()
        assert rate > 0.0

    def test_error_rate_by_component(self, tracker):
        tracker.record_error("embeddings", ValueError("e"))
        tracker.record_error("retrieval", ValueError("e"))

        emb_rate = tracker.get_error_rate(component="embeddings")
        all_rate = tracker.get_error_rate()
        assert emb_rate < all_rate

    def test_get_error_counts_by_type(self, tracker):
        tracker.record_error("api", ValueError("v1"))
        tracker.record_error("api", ValueError("v2"))
        tracker.record_error("api", RuntimeError("r1"))

        counts = tracker.get_error_counts_by_type()
        assert counts["ValueError"] == 2
        assert counts["RuntimeError"] == 1

    def test_get_error_counts_by_component(self, tracker):
        tracker.record_error("embeddings", ValueError("e"))
        tracker.record_error("embeddings", ValueError("e"))
        tracker.record_error("retrieval", ValueError("e"))

        counts = tracker.get_error_counts_by_component()
        assert counts["embeddings"] == 2
        assert counts["retrieval"] == 1

    def test_get_summary(self, tracker):
        tracker.record_error("api", ValueError("test"))
        summary = tracker.get_summary()

        assert summary["total_errors"] == 1
        assert "error_rate_per_minute" in summary
        assert "errors_by_component" in summary
        assert "errors_by_type" in summary
        assert "window_seconds" in summary

    def test_reset(self, tracker):
        tracker.record_error("api", ValueError("e"))
        tracker.reset()
        assert tracker.get_recent_errors() == []

    def test_clear_old_errors(self):
        # Use a very short window so errors expire immediately
        short_tracker = ErrorTracker(window_seconds=0)
        short_tracker.record_error("api", ValueError("old"))
        time.sleep(0.001)
        removed = short_tracker.clear_old_errors()
        assert removed >= 1


class TestGlobalErrorTracker:
    def test_get_error_tracker_returns_singleton(self):
        t1 = get_error_tracker()
        t2 = get_error_tracker()
        assert t1 is t2

    def test_record_error_global(self):
        # Should not raise
        record_error("test_component", ValueError("global test"), correlation_id="global-1")
