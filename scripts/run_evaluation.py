"""Run comprehensive evaluation on the RAG system.

This script evaluates retrieval quality using a gold dataset
and generates detailed reports.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.evaluation import (
    EvaluationReport,
    GoldDataset,
    RetrievalReportGenerator,
    aggregate_retrieval_metrics,
    evaluate_retrieval,
)
from src.embeddings.service import EmbeddingService
from src.vectorstore.chroma_store import ChromaStore


def main() -> None:
    """Run evaluation and generate report."""
    # Load configuration
    config = load_config()

    # Initialize services
    embedding_service = EmbeddingService(config)
    vector_store = ChromaStore(config)

    # Load gold dataset
    gold_dataset_path = Path(__file__).parent.parent / "tests" / "fixtures" / "gold_dataset.json"

    if not gold_dataset_path.exists():
        print(f"Error: Gold dataset not found at {gold_dataset_path}")
        print("Run scripts/build_gold_dataset.py first to create it.")
        return

    print(f"Loading gold dataset from {gold_dataset_path}")
    gold_dataset = GoldDataset.load(gold_dataset_path)
    print(f"Loaded {len(gold_dataset)} gold queries")
    print()

    # Evaluate each query
    print("Running evaluation...")
    all_metrics = []

    for i, gold_query in enumerate(gold_dataset, 1):
        print(f"[{i}/{len(gold_dataset)}] Evaluating: {gold_query.query[:60]}...")

        # Embed query
        query_embedding = embedding_service.embed_query(gold_query.query)

        # Search
        filters = {}
        if gold_query.language != "multi":
            filters["language"] = gold_query.language

        results = vector_store.search(
            query_embedding=query_embedding,
            top_k=10,
            filters=filters,
        )

        # Extract retrieved IDs
        retrieved_ids = [r["artifact"].id for r in results]

        # Evaluate
        metrics = evaluate_retrieval(gold_query, retrieved_ids)
        all_metrics.append(metrics)

        # Print per-query results
        print(f"  P@5: {metrics.precision_at_k[5]:.3f}, R@5: {metrics.recall_at_k[5]:.3f}, MRR: {metrics.mrr:.3f}")

    print()

    # Aggregate metrics
    aggregated = aggregate_retrieval_metrics(all_metrics)

    print("=== Aggregated Results ===")
    print(f"Queries evaluated: {len(gold_dataset)}")
    print()
    print("Precision@K:")
    for k, value in sorted(aggregated.precision_at_k.items()):
        print(f"  P@{k}: {value:.4f}")
    print()
    print("Recall@K:")
    for k, value in sorted(aggregated.recall_at_k.items()):
        print(f"  R@{k}: {value:.4f}")
    print()
    print(f"MRR: {aggregated.mrr:.4f}")
    print(f"MAP: {aggregated.map_score:.4f}")
    print(f"NDCG: {aggregated.ndcg:.4f}")
    print()

    # Generate report
    output_dir = Path(__file__).parent.parent / "data" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    report = RetrievalReportGenerator.generate_single_run_report(
        metrics=aggregated,
        title="CHEAP RAG Retrieval Evaluation",
        description=f"Evaluation on {len(gold_dataset)} gold queries",
        metadata={
            "num_queries": len(gold_dataset),
            "embedding_model": config.get("embeddings", {}).get("model_name", "unknown"),
            "vector_store": "ChromaDB",
        },
    )

    # Save as JSON and Markdown
    json_path = output_dir / "evaluation_report.json"
    md_path = output_dir / "evaluation_report.md"

    report.to_json(json_path)
    report.to_markdown(md_path)

    print(f"Report saved to:")
    print(f"  JSON: {json_path}")
    print(f"  Markdown: {md_path}")


if __name__ == "__main__":
    main()
