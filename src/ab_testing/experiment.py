"""A/B testing experiment framework for comparing variants.

This module provides infrastructure to run controlled experiments
comparing different embedding models, retrieval configurations, etc.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from src.ab_testing.variant import EmbeddingVariant, VariantConfig
from src.evaluation.gold_dataset import GoldDataset
from src.evaluation.metrics import (
    RetrievalMetrics,
    aggregate_retrieval_metrics,
    evaluate_retrieval,
)
from src.observability.logging import StructuredLogger


@dataclass
class ExperimentConfig:
    """Configuration for an A/B experiment.

    Attributes:
        name: Experiment name
        description: Experiment description
        variants: List of variant configurations to compare
        gold_dataset_path: Path to gold evaluation dataset
        vector_store_path: Path to vector store directory
        metadata: Additional experiment configuration
    """

    name: str
    description: str
    variants: list[VariantConfig]
    gold_dataset_path: str | Path
    vector_store_path: str | Path
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "variants": [v.to_dict() for v in self.variants],
            "gold_dataset_path": str(self.gold_dataset_path),
            "vector_store_path": str(self.vector_store_path),
            "metadata": self.metadata,
        }


@dataclass
class ExperimentResult:
    """Results from an A/B experiment.

    Attributes:
        experiment_name: Name of the experiment
        variant_name: Name of the variant
        metrics: Aggregated retrieval metrics
        per_query_metrics: Metrics for each individual query
        latency_stats: Latency statistics
        timestamp: When the experiment was run
        metadata: Additional result metadata
    """

    experiment_name: str
    variant_name: str
    metrics: RetrievalMetrics
    per_query_metrics: list[dict[str, Any]]
    latency_stats: dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "experiment_name": self.experiment_name,
            "variant_name": self.variant_name,
            "metrics": self.metrics.to_dict(),
            "per_query_metrics": self.per_query_metrics,
            "latency_stats": self.latency_stats,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class ABExperiment:
    """Run A/B experiments comparing different variants.

    Example:
        config = ExperimentConfig(
            name="embedding_model_comparison",
            variants=[BASELINE_VARIANT, BGE_LARGE_VARIANT],
            gold_dataset_path="tests/fixtures/gold_dataset.json",
            vector_store_path="data/ab_testing",
        )
        experiment = ABExperiment(config)
        results = experiment.run()
    """

    def __init__(self, config: ExperimentConfig):
        """Initialize experiment.

        Args:
            config: Experiment configuration
        """
        self.config = config
        self.logger = StructuredLogger("ab_testing")
        self._variants: dict[str, EmbeddingVariant] = {}

    def setup(self) -> None:
        """Set up experiment variants and index data."""
        self.logger.info(f"Setting up experiment: {self.config.name}")

        # Load gold dataset
        gold_dataset = GoldDataset.load(self.config.gold_dataset_path)
        self.logger.info(f"Loaded {len(gold_dataset)} gold queries")

        # Initialize variants
        for variant_config in self.config.variants:
            self.logger.info(f"Initializing variant: {variant_config.name}")

            variant = EmbeddingVariant(
                config=variant_config,
                vector_store_path=self.config.vector_store_path,
                collection_suffix=self.config.name,
            )
            variant.initialize()

            self._variants[variant_config.name] = variant

        self.logger.info("Experiment setup complete")

    def index_data(self, artifacts: list[Any]) -> None:
        """Index data in all variant vector stores.

        Args:
            artifacts: List of MetadataArtifact objects
        """
        self.logger.info(f"Indexing {len(artifacts)} artifacts in all variants")

        for variant_name, variant in self._variants.items():
            self.logger.info(f"Indexing variant: {variant_name}")
            variant.index_artifacts(artifacts)

        self.logger.info("Indexing complete")

    def run(self) -> dict[str, ExperimentResult]:
        """Run the experiment and collect results.

        Returns:
            Dictionary mapping variant name to results
        """
        import time

        self.logger.info(f"Running experiment: {self.config.name}")

        # Load gold dataset
        gold_dataset = GoldDataset.load(self.config.gold_dataset_path)

        results: dict[str, ExperimentResult] = {}

        # Run each variant
        for variant_name, variant in self._variants.items():
            self.logger.info(f"Evaluating variant: {variant_name}")

            per_query_metrics: list[dict[str, Any]] = []
            all_retrieval_metrics: list[RetrievalMetrics] = []
            latencies: list[float] = []

            # Evaluate on each gold query
            for gold_query in gold_dataset:
                start_time = time.perf_counter()

                # Search with variant
                search_results = variant.search(gold_query.query)

                # Record latency
                latency_ms = (time.perf_counter() - start_time) * 1000
                latencies.append(latency_ms)

                # Extract retrieved IDs
                retrieved_ids = [r["artifact"].id for r in search_results]

                # Evaluate retrieval quality
                retrieval_metrics = evaluate_retrieval(gold_query, retrieved_ids)
                all_retrieval_metrics.append(retrieval_metrics)

                # Record per-query metrics
                per_query_metrics.append(
                    {
                        "query_id": gold_query.id,
                        "query": gold_query.query,
                        "latency_ms": latency_ms,
                        "metrics": retrieval_metrics.to_dict(),
                        "num_results": len(retrieved_ids),
                    }
                )

            # Aggregate metrics
            aggregated_metrics = aggregate_retrieval_metrics(all_retrieval_metrics)

            # Calculate latency stats
            latency_stats = {
                "mean_ms": sum(latencies) / len(latencies) if latencies else 0.0,
                "median_ms": sorted(latencies)[len(latencies) // 2] if latencies else 0.0,
                "p95_ms": (sorted(latencies)[int(0.95 * len(latencies))] if latencies else 0.0),
                "min_ms": min(latencies) if latencies else 0.0,
                "max_ms": max(latencies) if latencies else 0.0,
            }

            # Create result
            result = ExperimentResult(
                experiment_name=self.config.name,
                variant_name=variant_name,
                metrics=aggregated_metrics,
                per_query_metrics=per_query_metrics,
                latency_stats=latency_stats,
                metadata={
                    "num_queries": len(gold_dataset),
                    "variant_config": variant.config.to_dict(),
                },
            )

            results[variant_name] = result

            self.logger.info(
                f"Variant {variant_name} complete",
                mean_precision_5=aggregated_metrics.precision_at_k.get(5, 0.0),
                mean_recall_5=aggregated_metrics.recall_at_k.get(5, 0.0),
                mrr=aggregated_metrics.mrr,
                mean_latency_ms=latency_stats["mean_ms"],
            )

        self.logger.info("Experiment complete")
        return results

    def save_results(
        self,
        results: dict[str, ExperimentResult],
        output_path: str | Path,
    ) -> None:
        """Save experiment results to JSON.

        Args:
            results: Experiment results
            output_path: Path to save results
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "experiment": self.config.to_dict(),
            "results": {name: result.to_dict() for name, result in results.items()},
            "timestamp": datetime.now().isoformat(),
        }

        output_path.write_text(json.dumps(data, indent=2))
        self.logger.info(f"Results saved to {output_path}")

    def cleanup(self) -> None:
        """Clean up experiment resources."""
        for variant in self._variants.values():
            variant.clear_index()

        self._variants.clear()


def run_embedding_comparison(
    baseline_model: str = "sentence-transformers/all-mpnet-base-v2",
    comparison_models: list[str] | None = None,
    gold_dataset_path: str | Path = "tests/fixtures/gold_dataset.json",
    artifacts: list[Any] | None = None,
    output_dir: str | Path = "data/ab_testing/results",
) -> dict[str, ExperimentResult]:
    """Run embedding model comparison experiment.

    Args:
        baseline_model: Baseline embedding model
        comparison_models: List of models to compare against baseline
        gold_dataset_path: Path to gold dataset
        artifacts: Optional list of artifacts to index (loads from index if None)
        output_dir: Directory to save results

    Returns:
        Dictionary mapping variant name to results
    """
    if comparison_models is None:
        comparison_models = ["BAAI/bge-large-en-v1.5"]

    # Create variant configs
    variants = [
        VariantConfig(
            name="baseline",
            embedding_model=baseline_model,
            embedding_dimension=768,
        )
    ]

    for i, model in enumerate(comparison_models):
        # Infer dimension from model name (heuristic)
        dimension = 1024 if "large" in model.lower() else 768

        variants.append(
            VariantConfig(
                name=f"variant_{i + 1}",
                embedding_model=model,
                embedding_dimension=dimension,
                metadata={"model": model},
            )
        )

    # Create experiment config
    config = ExperimentConfig(
        name=f"embedding_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        description="Comparison of embedding models for retrieval quality",
        variants=variants,
        gold_dataset_path=gold_dataset_path,
        vector_store_path="data/ab_testing/vector_stores",
    )

    # Run experiment
    experiment = ABExperiment(config)
    experiment.setup()

    if artifacts:
        experiment.index_data(artifacts)

    results = experiment.run()

    # Save results
    output_path = Path(output_dir) / f"{config.name}_results.json"
    experiment.save_results(results, output_path)

    experiment.cleanup()

    return results
