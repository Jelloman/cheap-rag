"""Tests for evaluation reporting infrastructure."""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from src.evaluation.metrics import RetrievalMetrics
from src.evaluation.reporting import (
    ABTestReportGenerator,
    EvaluationReport,
    RetrievalReportGenerator,
    generate_trend_report,
)


@pytest.fixture
def sample_metrics():
    return RetrievalMetrics(
        precision_at_k={1: 0.8, 5: 0.6, 10: 0.4},
        recall_at_k={1: 0.3, 5: 0.7, 10: 0.9},
        mrr=0.75,
        map_score=0.65,
        ndcg=0.7,
    )


@pytest.fixture
def sample_report(sample_metrics):
    return EvaluationReport(
        title="Test Report",
        description="A test evaluation report",
        timestamp=datetime(2026, 2, 19, 12, 0, 0),
        metrics=sample_metrics.to_dict(),
        metadata={"model": "all-mpnet-base-v2"},
    )


class TestEvaluationReport:
    def test_to_dict(self, sample_report):
        d = sample_report.to_dict()
        assert d["title"] == "Test Report"
        assert d["description"] == "A test evaluation report"
        assert "timestamp" in d
        assert "metrics" in d
        assert d["metadata"]["model"] == "all-mpnet-base-v2"

    def test_to_json(self, sample_report, tmp_path):
        path = tmp_path / "report.json"
        sample_report.to_json(path)

        assert path.exists()
        data = json.loads(path.read_text())
        assert data["title"] == "Test Report"
        assert "metrics" in data

    def test_to_json_creates_parent_dirs(self, sample_report, tmp_path):
        path = tmp_path / "subdir" / "deep" / "report.json"
        sample_report.to_json(path)
        assert path.exists()

    def test_to_markdown(self, sample_report, tmp_path):
        path = tmp_path / "report.md"
        sample_report.to_markdown(path)

        assert path.exists()
        content = path.read_text()
        assert "# Test Report" in content
        assert "A test evaluation report" in content
        assert "## Metrics" in content

    def test_to_markdown_includes_metadata(self, sample_report, tmp_path):
        path = tmp_path / "report.md"
        sample_report.to_markdown(path)

        content = path.read_text()
        assert "## Metadata" in content
        assert "all-mpnet-base-v2" in content

    def test_markdown_no_metadata(self, tmp_path):
        report = EvaluationReport(
            title="No Meta",
            description="desc",
            timestamp=datetime.now(),
            metrics={"mrr": 0.5},
        )
        path = tmp_path / "report.md"
        report.to_markdown(path)
        content = path.read_text()
        assert "# No Meta" in content
        # No metadata section when empty
        assert "## Metadata" not in content


class TestRetrievalReportGenerator:
    def test_generate_single_run_report(self, sample_metrics):
        report = RetrievalReportGenerator.generate_single_run_report(
            metrics=sample_metrics,
            title="Single Run",
            description="Baseline evaluation",
        )
        assert report.title == "Single Run"
        assert report.description == "Baseline evaluation"
        assert "mrr" in report.metrics

    def test_generate_single_run_with_metadata(self, sample_metrics):
        report = RetrievalReportGenerator.generate_single_run_report(
            metrics=sample_metrics,
            metadata={"run_id": "run-001"},
        )
        assert report.metadata["run_id"] == "run-001"

    def test_generate_comparison_report(self, sample_metrics):
        baseline = RetrievalMetrics(
            precision_at_k={5: 0.5},
            recall_at_k={5: 0.6},
            mrr=0.6,
        )
        comparison = RetrievalMetrics(
            precision_at_k={5: 0.7},
            recall_at_k={5: 0.8},
            mrr=0.8,
        )

        report = RetrievalReportGenerator.generate_comparison_report(
            baseline_metrics=baseline,
            comparison_metrics=comparison,
            baseline_name="Baseline",
            comparison_name="BGE-Large",
        )

        assert "Baseline" in report.metrics
        assert "BGE-Large" in report.metrics
        assert "improvements" in report.metrics

    def test_comparison_report_improvements(self):
        baseline = RetrievalMetrics(precision_at_k={5: 0.5}, mrr=0.5)
        comparison = RetrievalMetrics(precision_at_k={5: 0.75}, mrr=0.75)

        report = RetrievalReportGenerator.generate_comparison_report(
            baseline_metrics=baseline,
            comparison_metrics=comparison,
        )

        improvements = report.metrics["improvements"]
        # 0.75 / 0.5 - 1 = 50% improvement
        assert improvements["precision_at_5_improvement_%"] == pytest.approx(50.0)
        assert improvements["mrr_improvement_%"] == pytest.approx(50.0)

    def test_comparison_zero_baseline_mrr(self):
        baseline = RetrievalMetrics(mrr=0.0)
        comparison = RetrievalMetrics(mrr=0.5)

        # Should not raise even with zero baseline
        report = RetrievalReportGenerator.generate_comparison_report(
            baseline_metrics=baseline,
            comparison_metrics=comparison,
        )
        assert report.metrics["improvements"]["mrr_improvement_%"] == 0.0


class TestABTestReportGenerator:
    def test_generate_experiment_report_json(self, tmp_path):
        experiment_results = {
            "experiment": {
                "name": "embedding_test",
                "description": "Test A/B experiment",
            },
            "results": {
                "baseline": {"metrics": {"mrr": 0.6, "map": 0.5}},
                "bge-large": {"metrics": {"mrr": 0.8, "map": 0.7}},
            },
            "timestamp": "2026-02-19T12:00:00",
        }

        ABTestReportGenerator.generate_experiment_report(
            experiment_results=experiment_results,
            output_dir=tmp_path,
            format="json",
        )

        json_files = list(tmp_path.glob("*.json"))
        assert len(json_files) == 1
        data = json.loads(json_files[0].read_text())
        assert data["title"].startswith("A/B Test Results")

    def test_generate_experiment_report_markdown(self, tmp_path):
        experiment_results = {
            "experiment": {
                "name": "test_exp",
                "description": "Test",
            },
            "results": {
                "baseline": {"metrics": {"mrr": 0.6}},
            },
            "timestamp": "2026-02-19T12:00:00",
        }

        ABTestReportGenerator.generate_experiment_report(
            experiment_results=experiment_results,
            output_dir=tmp_path,
            format="markdown",
        )

        md_files = list(tmp_path.glob("*.md"))
        assert len(md_files) == 1

    def test_generate_experiment_report_both(self, tmp_path):
        experiment_results = {
            "experiment": {"name": "both_test", "description": "Both formats"},
            "results": {"v1": {"metrics": {"mrr": 0.5}}},
            "timestamp": "2026-02-19T12:00:00",
        }

        ABTestReportGenerator.generate_experiment_report(
            experiment_results=experiment_results,
            output_dir=tmp_path,
            format="both",
        )

        assert len(list(tmp_path.glob("*.json"))) == 1
        assert len(list(tmp_path.glob("*.md"))) == 1


class TestGenerateTrendReport:
    def test_generate_trend_report(self, tmp_path):
        # Create some fake result JSON files
        for i, val in enumerate([0.5, 0.6, 0.7]):
            result = {
                "timestamp": f"2026-02-1{i}T12:00:00",
                "metrics": {"precision_at_k": {5: val}, "recall_at_k": {5: val}, "mrr": val},
            }
            (tmp_path / f"run_{i:02d}.json").write_text(json.dumps(result))

        output = tmp_path / "trend.json"
        generate_trend_report(
            results_dir=tmp_path,
            output_path=output,
            metric_name="mrr",
        )

        assert output.exists()
        data = json.loads(output.read_text())
        assert data["metric"] == "mrr"
        assert len(data["data"]) == 3

    def test_generate_trend_report_empty_dir(self, tmp_path):
        output = tmp_path / "trend.json"
        generate_trend_report(results_dir=tmp_path, output_path=output)
        # With no JSON files, should return without creating output
        assert not output.exists()

    def test_trend_report_precision_at_k(self, tmp_path):
        # JSON serializes integer dict keys as strings ("5" not 5)
        result = {
            "timestamp": "2026-02-19T12:00:00",
            "metrics": {"precision_at_k": {"5": 0.65}, "recall_at_k": {"5": 0.7}, "mrr": 0.7},
        }
        (tmp_path / "run.json").write_text(json.dumps(result))

        output = tmp_path / "trend.json"
        generate_trend_report(
            results_dir=tmp_path,
            output_path=output,
            metric_name="precision_at_k",
            k=5,
        )

        data = json.loads(output.read_text())
        assert data["data"][0]["value"] == pytest.approx(0.65)
