"""Build gold evaluation dataset from test queries.

This script:
1. Loads test queries from tests/fixtures/test_queries.json
2. Runs each query through semantic search against the indexed vector store
3. Retrieves top-K candidates
4. Saves results for manual annotation

The output requires manual review to select truly relevant artifacts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.embeddings.service import EmbeddingService
from src.evaluation.gold_dataset import GoldDataset, GoldQuery
from src.vectorstore.chroma_store import ChromaVectorStore


def main() -> None:
    """Build gold dataset from test queries."""
    # Load configuration
    config = load_config()

    # Initialize embedding service
    embedding_service = EmbeddingService(
        model_name=config.embedding.model_name,
        device=config.embedding.device,
        cache_dir=config.embedding.cache_dir,
        batch_size=config.embedding.batch_size,
    )

    # Initialize vector store
    vector_store = ChromaVectorStore(
        persist_directory=config.vectorstore.persist_directory,
        collection_name=config.vectorstore.collection_name,
        distance_metric=config.vectorstore.distance_metric,
    )

    # Paths
    test_queries_path = Path(__file__).parent.parent / "tests" / "fixtures" / "test_queries.json"
    output_path = (
        Path(__file__).parent.parent / "tests" / "fixtures" / "gold_dataset.json"
    )

    print("Building gold dataset from test queries...")
    print(f"Input: {test_queries_path}")
    print(f"Output: {output_path}")
    print(f"Vector store items: {vector_store.count()}")
    print()

    # Load test queries
    test_data = json.loads(test_queries_path.read_text())
    top_k = 10

    gold_queries: list[GoldQuery] = []

    for query_data in test_data["queries"]:
        query_text = query_data["query"]
        query_id = query_data["id"]
        category = query_data["category"]
        language = query_data["language"]
        difficulty = query_data.get("difficulty", "medium")

        # Embed the query
        query_embedding = embedding_service.embed_text(query_text)

        # Build filters
        filters: dict[str, str] | None = None
        if language != "multi":
            filters = {"language": language}

        # Search vector store (returns ids, metadatas, distances)
        ids, metadatas, distances = vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters,
        )

        # Convert distances to similarities (cosine distance -> similarity)
        similarities = [1.0 - (d / 2.0) for d in distances]

        # Build display info for review
        candidates_info = []
        for artifact_id, metadata, similarity in zip(ids, metadatas, similarities):
            name = metadata.get("name", "unknown") if isinstance(metadata, dict) else "unknown"
            atype = metadata.get("type", "unknown") if isinstance(metadata, dict) else "unknown"
            candidates_info.append(f"{name} ({atype}) sim={similarity:.3f}")

        print(f"  [{query_id}] {query_text}")
        for i, info in enumerate(candidates_info[:5]):
            print(f"    {i+1}. {info}")

        # Create gold query with top-5 candidates as initial relevant set
        gold_query = GoldQuery(
            id=query_id,
            category=category,
            query=query_text,
            language=language,
            relevant_artifact_ids=list(ids[:5]),
            difficulty=difficulty,
            notes="Auto-generated from top candidates. Requires manual review.",
            metadata={
                "original_expected": query_data.get("expected_artifacts", []),
                "all_candidates": list(ids),
                "candidate_scores": similarities,
                "requires_manual_review": True,
            },
        )
        gold_queries.append(gold_query)

    dataset = GoldDataset(
        queries=gold_queries,
        description="Gold dataset for CHEAP RAG evaluation (Pagila database, auto-generated, needs review)",
        version="2.0",
        metadata={
            "generated_from": str(test_queries_path),
            "top_k": top_k,
            "requires_manual_review": True,
            "database": "pagila",
        },
    )

    dataset.save(output_path)

    print()
    print(f"Generated {len(dataset)} gold queries")
    print()
    print("IMPORTANT: This dataset requires manual review!")
    print("Review the output file and:")
    print("1. Verify that relevant_artifact_ids contains only truly relevant artifacts")
    print("2. Remove any false positives")
    print("3. Add any missing relevant artifacts")
    print("4. Set 'requires_manual_review' to false in each query's metadata")
    print()
    print(f"Dataset saved to: {output_path}")


if __name__ == "__main__":
    main()
