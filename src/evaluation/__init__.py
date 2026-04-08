"""Evaluation module for assessing retrieval and generation quality."""

from __future__ import annotations

from src.evaluation.gold_dataset import (
    ArtifactIdentifier,
    GoldDataset,
    GoldQuery,
    build_gold_dataset_from_index,
)
from src.evaluation.metrics import (
    EndToEndMetrics,
    GenerationMetrics,
    RetrievalMetrics,
    aggregate_retrieval_metrics,
    calculate_average_precision,
    calculate_mrr,
    calculate_ndcg,
    calculate_precision_at_k,
    calculate_recall_at_k,
    evaluate_retrieval,
)
from src.evaluation.reporting import (
    ABTestReportGenerator,
    EvaluationReport,
    RetrievalReportGenerator,
    generate_trend_report,
)

__all__ = [
    # Gold Dataset
    "ArtifactIdentifier",
    "GoldDataset",
    "GoldQuery",
    "build_gold_dataset_from_index",
    # Metrics
    "RetrievalMetrics",
    "GenerationMetrics",
    "EndToEndMetrics",
    "calculate_precision_at_k",
    "calculate_recall_at_k",
    "calculate_mrr",
    "calculate_average_precision",
    "calculate_ndcg",
    "evaluate_retrieval",
    "aggregate_retrieval_metrics",
    # Reporting
    "EvaluationReport",
    "RetrievalReportGenerator",
    "ABTestReportGenerator",
    "generate_trend_report",
]
