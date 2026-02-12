"""Run A/B test comparing different embedding models.

This script compares multiple embedding models side-by-side
to determine which provides better retrieval quality.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ab_testing import (
    ABExperiment,
    ExperimentConfig,
    VariantConfig,
)
from src.evaluation import ABTestReportGenerator


def main() -> None:
    """Run A/B test experiment."""
    # Define variants to test
    variants = [
        VariantConfig(
            name="baseline_mpnet",
            embedding_model="sentence-transformers/all-mpnet-base-v2",
            embedding_dimension=768,
            top_k=5,
            metadata={"description": "Current baseline model"},
        ),
        VariantConfig(
            name="bge_large",
            embedding_model="BAAI/bge-large-en-v1.5",
            embedding_dimension=1024,
            top_k=5,
            metadata={"description": "BGE Large - higher quality embeddings"},
        ),
        VariantConfig(
            name="bge_small",
            embedding_model="BAAI/bge-small-en-v1.5",
            embedding_dimension=384,
            top_k=5,
            metadata={"description": "BGE Small - faster, lower quality"},
        ),
    ]

    # Create experiment configuration
    config = ExperimentConfig(
        name="embedding_model_comparison_2026",
        description="Comparison of sentence-transformers vs BGE models for metadata retrieval",
        variants=variants,
        gold_dataset_path="tests/fixtures/gold_dataset.json",
        vector_store_path="data/ab_testing/vector_stores",
        metadata={
            "objective": "Determine if BGE models improve retrieval quality over current baseline",
            "hypothesis": "BGE Large will improve P@5 by at least 10%",
        },
    )

    print(f"Running experiment: {config.name}")
    print(f"Description: {config.description}")
    print(f"Variants: {len(config.variants)}")
    print()

    # Check if gold dataset exists
    gold_path = Path(config.gold_dataset_path)
    if not gold_path.exists():
        print(f"Error: Gold dataset not found at {gold_path}")
        print("Run scripts/build_gold_dataset.py first to create it.")
        return

    # Create and run experiment
    experiment = ABExperiment(config)

    print("Setting up experiment...")
    experiment.setup()

    print("Note: This script assumes artifacts are already indexed in the baseline vector store.")
    print("If you need to index artifacts, use scripts/index_metadata.py first.")
    print()

    print("Running experiment (this may take a while)...")
    results = experiment.run()

    print()
    print("=== Experiment Results ===")
    print()

    for variant_name, result in results.items():
        print(f"Variant: {variant_name}")
        print(f"  P@5: {result.metrics.precision_at_k[5]:.4f}")
        print(f"  R@5: {result.metrics.recall_at_k[5]:.4f}")
        print(f"  MRR: {result.metrics.mrr:.4f}")
        print(f"  Mean Latency: {result.latency_stats['mean_ms']:.2f}ms")
        print()

    # Save results
    output_dir = Path("data/ab_testing/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / f"{config.name}_results.json"
    experiment.save_results(results, results_path)

    # Generate report
    import json

    experiment_data = json.loads(results_path.read_text())
    ABTestReportGenerator.generate_experiment_report(
        experiment_results=experiment_data,
        output_dir=output_dir,
        format="both",
    )

    print(f"Results saved to: {results_path}")
    print(f"Report generated in: {output_dir}")

    # Cleanup
    print()
    print("Cleaning up experiment resources...")
    experiment.cleanup()
    print("Done!")


if __name__ == "__main__":
    main()
