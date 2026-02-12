"""Evaluation reporting and visualization tools.

This module provides tools to generate reports and visualizations
for retrieval and generation quality metrics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from src.evaluation.metrics import RetrievalMetrics


@dataclass
class EvaluationReport:
    """Comprehensive evaluation report.

    Attributes:
        title: Report title
        description: Report description
        timestamp: When the report was generated
        metrics: Evaluation metrics
        metadata: Additional report metadata
    """

    title: str
    description: str
    timestamp: datetime
    metrics: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "title": self.title,
            "description": self.description,
            "timestamp": self.timestamp.isoformat(),
            "metrics": self.metrics,
            "metadata": self.metadata,
        }

    def to_json(self, path: str | Path) -> None:
        """Save report as JSON.

        Args:
            path: Path to save JSON file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    def to_markdown(self, path: str | Path) -> None:
        """Generate markdown report.

        Args:
            path: Path to save markdown file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        content = self._generate_markdown()
        path.write_text(content)

    def _generate_markdown(self) -> str:
        """Generate markdown content for report.

        Returns:
            Markdown formatted report
        """
        lines = [
            f"# {self.title}",
            "",
            self.description,
            "",
            f"**Generated:** {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]

        # Add metadata if present
        if self.metadata:
            lines.extend(
                [
                    "## Metadata",
                    "",
                ]
            )
            for key, value in self.metadata.items():
                lines.append(f"- **{key}:** {value}")
            lines.append("")

        # Add metrics
        lines.extend(
            [
                "## Metrics",
                "",
            ]
        )

        # Format metrics recursively
        lines.extend(self._format_metrics_markdown(self.metrics))

        return "\n".join(lines)

    def _format_metrics_markdown(
        self,
        metrics: dict[str, Any],
        level: int = 3,
    ) -> list[str]:
        """Format metrics as markdown recursively.

        Args:
            metrics: Metrics dictionary
            level: Header level

        Returns:
            List of markdown lines
        """
        lines: list[str] = []

        for key, value in metrics.items():
            if isinstance(value, dict):
                lines.append(f"{'#' * level} {key.replace('_', ' ').title()}")
                lines.append("")
                lines.extend(self._format_metrics_markdown(value, level + 1))
            elif isinstance(value, (list, tuple)):
                lines.append(f"- **{key}:** {', '.join(str(v) for v in value)}")
            else:
                # Format numbers nicely
                formatted_value = f"{value:.4f}" if isinstance(value, float) else str(value)
                lines.append(f"- **{key}:** {formatted_value}")

        lines.append("")
        return lines


class RetrievalReportGenerator:
    """Generate reports for retrieval evaluation."""

    @staticmethod
    def generate_single_run_report(
        metrics: RetrievalMetrics,
        title: str = "Retrieval Evaluation Report",
        description: str = "Evaluation of retrieval quality",
        metadata: dict[str, Any] | None = None,
    ) -> EvaluationReport:
        """Generate report for a single evaluation run.

        Args:
            metrics: Retrieval metrics
            title: Report title
            description: Report description
            metadata: Additional metadata

        Returns:
            EvaluationReport instance
        """
        return EvaluationReport(
            title=title,
            description=description,
            timestamp=datetime.now(),
            metrics=metrics.to_dict(),
            metadata=metadata or {},
        )

    @staticmethod
    def generate_comparison_report(
        baseline_metrics: RetrievalMetrics,
        comparison_metrics: RetrievalMetrics,
        baseline_name: str = "Baseline",
        comparison_name: str = "Comparison",
        title: str = "Retrieval Comparison Report",
        description: str = "Side-by-side comparison of retrieval methods",
        metadata: dict[str, Any] | None = None,
    ) -> EvaluationReport:
        """Generate comparison report for two evaluation runs.

        Args:
            baseline_metrics: Baseline retrieval metrics
            comparison_metrics: Comparison retrieval metrics
            baseline_name: Name for baseline
            comparison_name: Name for comparison
            title: Report title
            description: Report description
            metadata: Additional metadata

        Returns:
            EvaluationReport instance
        """
        # Calculate improvements
        improvements = {}
        for k in baseline_metrics.precision_at_k:
            baseline_p = baseline_metrics.precision_at_k[k]
            comparison_p = comparison_metrics.precision_at_k[k]

            improvement = (comparison_p - baseline_p) / baseline_p * 100 if baseline_p > 0 else 0.0

            improvements[f"precision_at_{k}_improvement_%"] = improvement

        # Similar for recall
        for k in baseline_metrics.recall_at_k:
            baseline_r = baseline_metrics.recall_at_k[k]
            comparison_r = comparison_metrics.recall_at_k[k]

            improvement = (comparison_r - baseline_r) / baseline_r * 100 if baseline_r > 0 else 0.0

            improvements[f"recall_at_{k}_improvement_%"] = improvement

        # MRR improvement
        if baseline_metrics.mrr > 0:
            improvements["mrr_improvement_%"] = (
                (comparison_metrics.mrr - baseline_metrics.mrr) / baseline_metrics.mrr
            ) * 100
        else:
            improvements["mrr_improvement_%"] = 0.0

        report_metrics = {
            baseline_name: baseline_metrics.to_dict(),
            comparison_name: comparison_metrics.to_dict(),
            "improvements": improvements,
        }

        return EvaluationReport(
            title=title,
            description=description,
            timestamp=datetime.now(),
            metrics=report_metrics,
            metadata=metadata or {},
        )


class ABTestReportGenerator:
    """Generate reports for A/B test results."""

    @staticmethod
    def generate_experiment_report(
        experiment_results: dict[str, Any],
        output_dir: str | Path,
        format: str = "both",
    ) -> None:
        """Generate report from A/B experiment results.

        Args:
            experiment_results: Experiment results dictionary
            output_dir: Directory to save reports
            format: Report format ("json", "markdown", or "both")
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        experiment_name = experiment_results["experiment"]["name"]
        timestamp = experiment_results["timestamp"]

        # Extract metrics for each variant
        variant_metrics = {}
        for variant_name, result in experiment_results["results"].items():
            variant_metrics[variant_name] = result["metrics"]

        # Create comparison report
        report = EvaluationReport(
            title=f"A/B Test Results: {experiment_name}",
            description=experiment_results["experiment"]["description"],
            timestamp=datetime.fromisoformat(timestamp),
            metrics={"variants": variant_metrics},
            metadata={
                "experiment_name": experiment_name,
                "num_variants": len(variant_metrics),
            },
        )

        # Save in requested format(s)
        base_name = f"{experiment_name}_report"

        if format in ("json", "both"):
            report.to_json(output_dir / f"{base_name}.json")

        if format in ("markdown", "both"):
            report.to_markdown(output_dir / f"{base_name}.md")


def generate_trend_report(
    results_dir: str | Path,
    output_path: str | Path,
    metric_name: str = "precision_at_k",
    k: int = 5,
) -> None:
    """Generate trend report from multiple evaluation runs.

    Args:
        results_dir: Directory containing result JSON files
        output_path: Path to save trend report
        metric_name: Metric to track over time
        k: K value for @K metrics
    """
    results_dir = Path(results_dir)
    output_path = Path(output_path)

    # Load all result files
    results: list[dict[str, Any]] = []
    for json_file in sorted(results_dir.glob("*.json")):
        data = json.loads(json_file.read_text())
        results.append(data)

    if not results:
        return

    # Extract metric values over time
    trend_data: list[dict[str, Any]] = []
    for result in results:
        timestamp = result.get("timestamp", "")
        metrics = result.get("metrics", {})

        if metric_name == "precision_at_k":
            value = metrics.get("precision_at_k", {}).get(k, 0.0)
        elif metric_name == "recall_at_k":
            value = metrics.get("recall_at_k", {}).get(k, 0.0)
        elif metric_name == "mrr":
            value = metrics.get("mrr", 0.0)
        elif metric_name == "map":
            value = metrics.get("map", 0.0)
        else:
            value = 0.0

        trend_data.append({"timestamp": timestamp, "value": value})

    # Create trend report
    report = {
        "title": f"Trend Report: {metric_name}",
        "metric": metric_name,
        "k": k if "@k" in metric_name else None,
        "data": trend_data,
        "generated": datetime.now().isoformat(),
    }

    output_path.write_text(json.dumps(report, indent=2))
