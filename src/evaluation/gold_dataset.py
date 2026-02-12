"""Gold dataset with ground truth annotations for evaluation.

This module provides tools to create and manage gold question datasets
with known-relevant artifact IDs for quantitative retrieval evaluation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class GoldQuery:
    """A test query with ground truth relevant artifacts.

    Attributes:
        id: Unique query identifier
        category: Query category (entity_lookup, relationship_query, etc.)
        query: The natural language question
        language: Target language filter
        relevant_artifact_ids: Ground truth list of relevant artifact IDs
        difficulty: Query difficulty (easy, medium, hard)
        notes: Optional notes about expected behavior
    """

    id: str
    category: str
    query: str
    language: str
    relevant_artifact_ids: list[str]
    difficulty: str = "medium"
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "category": self.category,
            "query": self.query,
            "language": self.language,
            "relevant_artifact_ids": self.relevant_artifact_ids,
            "difficulty": self.difficulty,
            "notes": self.notes,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GoldQuery:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            category=data["category"],
            query=data["query"],
            language=data["language"],
            relevant_artifact_ids=data.get("relevant_artifact_ids", []),
            difficulty=data.get("difficulty", "medium"),
            notes=data.get("notes", ""),
            metadata=data.get("metadata", {}),
        )


@dataclass
class GoldDataset:
    """Collection of gold queries for evaluation.

    Attributes:
        queries: List of gold queries
        description: Dataset description
        version: Dataset version string
        metadata: Additional metadata
    """

    queries: list[GoldQuery]
    description: str = "Gold dataset for CHEAP RAG evaluation"
    version: str = "1.0"
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        """Save dataset to JSON file."""
        path = Path(path)
        data = {
            "description": self.description,
            "version": self.version,
            "metadata": self.metadata,
            "queries": [q.to_dict() for q in self.queries],
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> GoldDataset:
        """Load dataset from JSON file."""
        path = Path(path)
        data = json.loads(path.read_text())

        return cls(
            queries=[GoldQuery.from_dict(q) for q in data["queries"]],
            description=data.get("description", ""),
            version=data.get("version", "1.0"),
            metadata=data.get("metadata", {}),
        )

    def filter_by_category(self, category: str) -> list[GoldQuery]:
        """Get all queries in a specific category."""
        return [q for q in self.queries if q.category == category]

    def filter_by_language(self, language: str) -> list[GoldQuery]:
        """Get all queries for a specific language."""
        return [q for q in self.queries if q.language == language]

    def filter_by_difficulty(self, difficulty: str) -> list[GoldQuery]:
        """Get all queries of a specific difficulty level."""
        return [q for q in self.queries if q.difficulty == difficulty]

    def __len__(self) -> int:
        """Get number of queries in dataset."""
        return len(self.queries)

    def __iter__(self):
        """Iterate over queries."""
        return iter(self.queries)


def build_gold_dataset_from_index(
    vector_store: Any,
    test_queries_path: str | Path,
    output_path: str | Path,
    top_k: int = 10,
) -> GoldDataset:
    """Build gold dataset by running test queries and manually annotating results.

    This function:
    1. Loads the existing test queries
    2. Runs each query through the vector store
    3. Retrieves top-K candidates
    4. Saves results for manual annotation

    Args:
        vector_store: Initialized ChromaVectorStore instance
        test_queries_path: Path to test_queries.json
        output_path: Path to save gold dataset
        top_k: Number of candidates to retrieve per query

    Returns:
        GoldDataset with candidate artifact IDs (requires manual review)
    """
    test_queries_path = Path(test_queries_path)
    test_data = json.loads(test_queries_path.read_text())

    from src.config import load_config
    from src.embeddings.service import EmbeddingService

    config = load_config()
    embedding_service = EmbeddingService(
        model_name=config.embedding.model_name,
        device=config.embedding.device,
        cache_dir=config.embedding.cache_dir,
        batch_size=config.embedding.batch_size,
    )

    gold_queries: list[GoldQuery] = []

    for query_data in test_data["queries"]:
        query_text = query_data["query"]
        query_id = query_data["id"]
        category = query_data["category"]
        language = query_data["language"]
        difficulty = query_data.get("difficulty", "medium")

        # Embed the query
        query_embedding = embedding_service.embed_query(query_text)

        # Search vector store
        filters = {}
        if language != "multi":
            filters["language"] = language

        results = vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters,
        )

        # Extract artifact IDs from top results
        candidate_ids = [r["artifact"].id for r in results]

        # Create gold query with candidates (manual review needed)
        gold_query = GoldQuery(
            id=query_id,
            category=category,
            query=query_text,
            language=language,
            relevant_artifact_ids=candidate_ids[:5],  # Top 5 as initial candidates
            difficulty=difficulty,
            notes=f"Auto-generated from top {top_k} candidates. Requires manual review.",
            metadata={
                "original_expected": query_data.get("expected_artifacts", []),
                "all_candidates": candidate_ids,
                "candidate_scores": [r["similarity"] for r in results],
            },
        )
        gold_queries.append(gold_query)

    dataset = GoldDataset(
        queries=gold_queries,
        description="Gold dataset for CHEAP RAG evaluation (auto-generated, needs review)",
        version="2.0",
        metadata={
            "generated_from": str(test_queries_path),
            "top_k": top_k,
            "requires_manual_review": True,
        },
    )

    dataset.save(output_path)
    return dataset
