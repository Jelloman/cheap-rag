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
from src.evaluation.gold_dataset import ArtifactIdentifier, GoldDataset, GoldQuery, compute_artifact_component
from src.observability.tracing import init_tracing
from src.vectorstore.chroma_store import ChromaVectorStore


def main() -> None:
    """Build gold dataset from test queries."""
    # Disable console span exporter — trace JSON on stdout is too noisy for a script
    init_tracing(enable_console=False)

    # Load configuration
    config = load_config()

    # Initialize embedding service
    embedding_service = EmbeddingService(
        model_name=config.embedding.model_name,
        device=config.embedding.device,
        cache_dir=config.embedding.cache_dir,
        batch_size=config.embedding.batch_size,
        local_files_only=config.embedding.local_files_only,
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

        # Group top-5 candidates by type into relevant_artifacts; record all-K as candidates
        relevant_artifacts: dict[str, list[ArtifactIdentifier]] = {}
        all_candidates: list[dict[str, object]] = []
        for i, (artifact_id, meta) in enumerate(zip(ids, metadatas)):
            artifact_type = str(meta.get("type", "unknown"))
            component = compute_artifact_component(meta) or None
            ident = ArtifactIdentifier(
                name=str(meta.get("name", "")),
                component=component,
                artifact_id=artifact_id,
            )
            all_candidates.append({**ident.to_dict(), "type": artifact_type, "similarity": similarities[i]})
            if i < 5:
                relevant_artifacts.setdefault(artifact_type, []).append(ident)

        print(f"  [{query_id}] {query_text}")
        for i, cand in enumerate(all_candidates[:5]):
            print(f"    {i+1}. {cand['name']} ({cand['type']}) sim={similarities[i]:.3f}")

        # Create gold query with top-5 candidates as initial relevant set
        gold_query = GoldQuery(
            id=query_id,
            category=category,
            query=query_text,
            language=language,
            relevant_artifacts=relevant_artifacts,
            difficulty=difficulty,
            notes="Auto-generated from top candidates. Requires manual review.",
            metadata={
                "original_expected": query_data.get("expected_artifacts", []),
                "all_candidates": all_candidates,
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
    print("1. Verify that relevant_artifacts contains only truly relevant artifacts")
    print("2. Remove any false positives")
    print("3. Add any missing relevant artifacts")
    print("4. Set 'requires_manual_review' to false in each query's metadata")
    print()
    print(f"Dataset saved to: {output_path}")


if __name__ == "__main__":
    main()
