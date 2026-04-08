"""Evaluation metrics for retrieval and generation quality.

This module provides quantitative metrics for assessing:
- Retrieval quality (Precision, Recall, MRR, NDCG)
- Generation quality (BLEU, ROUGE, citation accuracy)
- End-to-end performance
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.evaluation.gold_dataset import GoldQuery


@dataclass
class RetrievalMetrics:
    """Metrics for evaluating retrieval quality.

    Attributes:
        precision_at_k: Precision@K for various K values
        recall_at_k: Recall@K for various K values
        mrr: Mean Reciprocal Rank
        map_score: Mean Average Precision
        ndcg: Normalized Discounted Cumulative Gain
        metadata: Additional metric metadata
    """

    precision_at_k: dict[int, float] = field(default_factory=dict)
    recall_at_k: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    map_score: float = 0.0
    ndcg: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "precision_at_k": self.precision_at_k,
            "recall_at_k": self.recall_at_k,
            "mrr": self.mrr,
            "map": self.map_score,
            "ndcg": self.ndcg,
            "metadata": self.metadata,
        }


@dataclass
class GenerationMetrics:
    """Metrics for evaluating generation quality.

    Attributes:
        citation_precision: Fraction of citations that are valid
        citation_recall: Fraction of relevant docs that are cited
        answer_relevance: Manual/auto assessment of answer relevance (0-1)
        answer_correctness: Manual assessment of factual correctness (0-1)
        answer_completeness: Manual assessment of completeness (0-1)
        has_answer: Whether the system provided an answer (vs "I don't know")
        metadata: Additional metric metadata
    """

    citation_precision: float = 0.0
    citation_recall: float = 0.0
    answer_relevance: float = 0.0
    answer_correctness: float = 0.0
    answer_completeness: float = 0.0
    has_answer: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "citation_precision": self.citation_precision,
            "citation_recall": self.citation_recall,
            "answer_relevance": self.answer_relevance,
            "answer_correctness": self.answer_correctness,
            "answer_completeness": self.answer_completeness,
            "has_answer": self.has_answer,
            "metadata": self.metadata,
        }


@dataclass
class EndToEndMetrics:
    """Combined metrics for full pipeline evaluation.

    Attributes:
        retrieval: Retrieval metrics
        generation: Generation metrics
        latency_ms: End-to-end latency in milliseconds
        num_queries: Number of queries evaluated
        metadata: Additional metric metadata
    """

    retrieval: RetrievalMetrics
    generation: GenerationMetrics
    latency_ms: float = 0.0
    num_queries: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "retrieval": self.retrieval.to_dict(),
            "generation": self.generation.to_dict(),
            "latency_ms": self.latency_ms,
            "num_queries": self.num_queries,
            "metadata": self.metadata,
        }


def calculate_precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Calculate Precision@K.

    Precision@K = (# relevant items in top K) / K

    Args:
        retrieved_ids: List of retrieved artifact IDs in rank order
        relevant_ids: Set of ground truth relevant artifact IDs
        k: Number of top results to consider

    Returns:
        Precision@K score (0.0 to 1.0)
    """
    if k <= 0 or not retrieved_ids:
        return 0.0

    top_k = retrieved_ids[:k]
    relevant_in_top_k = sum(1 for item_id in top_k if item_id in relevant_ids)
    return relevant_in_top_k / k


def calculate_recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Calculate Recall@K.

    Recall@K = (# relevant items in top K) / (total # relevant items)

    Args:
        retrieved_ids: List of retrieved artifact IDs in rank order
        relevant_ids: Set of ground truth relevant artifact IDs
        k: Number of top results to consider

    Returns:
        Recall@K score (0.0 to 1.0)
    """
    if not relevant_ids or k <= 0:
        return 0.0

    top_k = retrieved_ids[:k]
    relevant_in_top_k = sum(1 for item_id in top_k if item_id in relevant_ids)
    return relevant_in_top_k / len(relevant_ids)


def calculate_mrr(
    retrieved_ids: list[str],
    relevant_ids: set[str],
) -> float:
    """Calculate Mean Reciprocal Rank for a single query.

    MRR = 1 / (rank of first relevant item)

    Args:
        retrieved_ids: List of retrieved artifact IDs in rank order
        relevant_ids: Set of ground truth relevant artifact IDs

    Returns:
        Reciprocal rank (0.0 to 1.0)
    """
    for rank, item_id in enumerate(retrieved_ids, start=1):
        if item_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def calculate_average_precision(
    retrieved_ids: list[str],
    relevant_ids: set[str],
) -> float:
    """Calculate Average Precision for a single query.

    AP = (sum of P@k for each relevant item) / (total # relevant items)

    Args:
        retrieved_ids: List of retrieved artifact IDs in rank order
        relevant_ids: Set of ground truth relevant artifact IDs

    Returns:
        Average Precision score (0.0 to 1.0)
    """
    if not relevant_ids:
        return 0.0

    precisions: list[float] = []
    num_relevant_seen = 0

    for k, item_id in enumerate(retrieved_ids, start=1):
        if item_id in relevant_ids:
            num_relevant_seen += 1
            precision_at_k = num_relevant_seen / k
            precisions.append(precision_at_k)

    if not precisions:
        return 0.0

    return sum(precisions) / len(relevant_ids)


def calculate_ndcg(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int | None = None,
) -> float:
    """Calculate Normalized Discounted Cumulative Gain.

    NDCG measures ranking quality with position discount.
    Assumes binary relevance (relevant=1, not relevant=0).

    Args:
        retrieved_ids: List of retrieved artifact IDs in rank order
        relevant_ids: Set of ground truth relevant artifact IDs
        k: Number of top results to consider (None = all)

    Returns:
        NDCG score (0.0 to 1.0)
    """
    import math

    if not relevant_ids:
        return 0.0

    # Truncate to top-k if specified
    if k is not None:
        retrieved_ids = retrieved_ids[:k]

    # Calculate DCG
    dcg = 0.0
    for i, item_id in enumerate(retrieved_ids):
        if item_id in relevant_ids:
            # Binary relevance: rel = 1 if relevant, 0 otherwise
            # DCG = sum(rel_i / log2(i+1))
            dcg += 1.0 / math.log2(i + 2)  # i+2 because enumerate is 0-indexed

    # Calculate IDCG (ideal DCG)
    num_relevant = min(len(relevant_ids), len(retrieved_ids))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(num_relevant))

    if idcg == 0:
        return 0.0

    return dcg / idcg


def evaluate_retrieval(
    gold_query: GoldQuery,
    retrieved_ids: list[str],
    k_values: list[int] | None = None,
) -> RetrievalMetrics:
    """Evaluate retrieval results for a single query.

    Args:
        gold_query: Gold query with ground truth relevant IDs
        retrieved_ids: List of retrieved artifact IDs in rank order
        k_values: K values for Precision@K and Recall@K (default: [1, 3, 5, 10])

    Returns:
        RetrievalMetrics with calculated scores
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    relevant_ids = gold_query.relevant_ids

    metrics = RetrievalMetrics()

    # Calculate Precision@K and Recall@K
    for k in k_values:
        metrics.precision_at_k[k] = calculate_precision_at_k(retrieved_ids, relevant_ids, k)
        metrics.recall_at_k[k] = calculate_recall_at_k(retrieved_ids, relevant_ids, k)

    # Calculate MRR
    metrics.mrr = calculate_mrr(retrieved_ids, relevant_ids)

    # Calculate MAP (Average Precision for single query)
    metrics.map_score = calculate_average_precision(retrieved_ids, relevant_ids)

    # Calculate NDCG
    metrics.ndcg = calculate_ndcg(retrieved_ids, relevant_ids, k=10)

    metrics.metadata = {
        "query_id": gold_query.id,
        "num_relevant": len(relevant_ids),
        "num_retrieved": len(retrieved_ids),
    }

    return metrics


def aggregate_retrieval_metrics(
    metrics_list: list[RetrievalMetrics],
) -> RetrievalMetrics:
    """Aggregate retrieval metrics across multiple queries.

    Args:
        metrics_list: List of RetrievalMetrics from individual queries

    Returns:
        Aggregated RetrievalMetrics with mean scores
    """
    if not metrics_list:
        return RetrievalMetrics()

    n = len(metrics_list)

    # Aggregate Precision@K and Recall@K
    all_k_values: set[int] = set()
    for m in metrics_list:
        all_k_values.update(m.precision_at_k.keys())

    aggregated = RetrievalMetrics()

    for k in all_k_values:
        precisions = [m.precision_at_k.get(k, 0.0) for m in metrics_list]
        recalls = [m.recall_at_k.get(k, 0.0) for m in metrics_list]

        aggregated.precision_at_k[k] = sum(precisions) / n
        aggregated.recall_at_k[k] = sum(recalls) / n

    # Aggregate MRR
    aggregated.mrr = sum(m.mrr for m in metrics_list) / n

    # Aggregate MAP
    aggregated.map_score = sum(m.map_score for m in metrics_list) / n

    # Aggregate NDCG
    aggregated.ndcg = sum(m.ndcg for m in metrics_list) / n

    aggregated.metadata = {
        "num_queries": n,
        "aggregation": "mean",
    }

    return aggregated
