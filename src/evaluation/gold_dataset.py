"""Gold dataset with ground truth annotations for evaluation.

This module provides tools to create and manage gold question datasets
with known-relevant artifact IDs for quantitative retrieval evaluation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Postgres artifact types whose component is the owning table (stored in table_name metadata)
_PG_TABLE_SCOPED: frozenset[str] = frozenset({"column", "index", "constraint", "trigger"})


def compute_artifact_component(meta: dict[str, Any]) -> str:
    """Derive a human-readable component grouping key from artifact metadata.

    Java: plain class/interface name — source filename stem (no path, no .java).
    PostgreSQL:
      - column / index / constraint / trigger → owning table name (table_name field)
      - relationship → from_table field
      - table / view / function / sequence / … → object name itself
    All values are schema-free as stored by the extractors.

    Args:
        meta: Flat metadata dict as stored in (or read from) ChromaDB.

    Returns:
        Component string, or empty string when not derivable.
    """
    language = str(meta.get("language", ""))
    artifact_type = str(meta.get("type", ""))

    if language == "java":
        source_file = str(meta.get("source_file", ""))
        return Path(source_file).stem if source_file else ""

    if language == "postgresql":
        if artifact_type in _PG_TABLE_SCOPED:
            return str(meta.get("table_name", ""))
        if artifact_type == "relationship":
            return str(meta.get("from_table", ""))
        return str(meta.get("name", ""))

    return ""


@dataclass
class ArtifactIdentifier:
    """Minimal human-readable identifier for an artifact in a gold dataset entry.

    Designed for ergonomic hand-editing:
    - Top-level artifacts (table, view, class, interface) need only ``name``.
    - Child artifacts (column, field, method) also need ``component`` — the
      owning table name (PostgreSQL) or class name (Java).
    - ``artifact_id`` is optional. When omitted, the evaluation runner resolves
      it via a ChromaDB lookup before computing metrics.

    Attributes:
        name: Artifact name (e.g. "film", "customer_id", "Catalog")
        component: Owning table (Postgres child types) or class name (Java members)
        artifact_id: Pre-resolved ChromaDB ID; may be omitted in hand-authored entries
    """

    name: str
    component: str | None = None
    artifact_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict, omitting None fields."""
        d: dict[str, Any] = {"name": self.name}
        if self.component:
            d["component"] = self.component
        if self.artifact_id:
            d["artifact_id"] = self.artifact_id
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArtifactIdentifier:
        """Deserialize from a dict."""
        return cls(
            name=data.get("name", ""),
            component=data.get("component") or None,
            artifact_id=data.get("artifact_id") or None,
        )


@dataclass
class GoldQuery:
    """A test query with ground truth relevant artifacts.

    Ground truth is stored as ``relevant_artifacts``: a dict keyed by artifact type
    (e.g. ``"table"``, ``"column"``, ``"class"``) whose values are lists of
    :class:`ArtifactIdentifier` objects.  This makes the JSON easy to hand-edit —
    top-level objects need only a ``name``; child objects (columns, fields, methods)
    also need a ``component`` (owning table or class).  The ``artifact_id`` field
    is optional and is resolved at evaluation time when absent.

    Attributes:
        id: Unique query identifier
        category: Query category (entity_lookup, relationship_query, etc.)
        query: The natural language question
        language: Target language filter
        relevant_artifacts: Ground truth, keyed by artifact type
        difficulty: Query difficulty (easy, medium, hard)
        notes: Optional notes about expected behavior
    """

    id: str
    category: str
    query: str
    language: str
    relevant_artifacts: dict[str, list[ArtifactIdentifier]]
    difficulty: str = "medium"
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def relevant_ids(self) -> set[str]:
        """Return the set of pre-resolved artifact IDs for metric computation.

        Only includes identifiers whose ``artifact_id`` has been populated.
        Call the evaluation runner's ``resolve_artifact_ids()`` first to fill
        in any entries that were hand-authored without an explicit ID.
        """
        return {
            ident.artifact_id
            for idents in self.relevant_artifacts.values()
            for ident in idents
            if ident.artifact_id
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "category": self.category,
            "query": self.query,
            "language": self.language,
            "relevant_artifacts": {
                artifact_type: [i.to_dict() for i in idents]
                for artifact_type, idents in self.relevant_artifacts.items()
            },
            "difficulty": self.difficulty,
            "notes": self.notes,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GoldQuery:
        """Create from dictionary."""
        relevant_artifacts: dict[str, list[ArtifactIdentifier]] = {
            artifact_type: [ArtifactIdentifier.from_dict(i) for i in idents]
            for artifact_type, idents in data.get("relevant_artifacts", {}).items()
        }

        return cls(
            id=data["id"],
            category=data["category"],
            query=data["query"],
            language=data["language"],
            relevant_artifacts=relevant_artifacts,
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
        local_files_only=config.embedding.local_files_only,
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

        # results is (ids, metadatas, distances)
        result_ids, metadatas, distances = results  # type: ignore[misc]
        similarities = [1.0 - (d / 2.0) for d in distances]

        # Group top-5 by type into relevant_artifacts; record all-K as candidates
        relevant_artifacts: dict[str, list[ArtifactIdentifier]] = {}
        all_candidates: list[dict[str, Any]] = []
        for i, (artifact_id, meta) in enumerate(zip(result_ids, metadatas, strict=False)):
            artifact_type = str(meta.get("type", "unknown"))
            component = compute_artifact_component(meta) or None
            ident = ArtifactIdentifier(
                name=str(meta.get("name", "")),
                component=component,
                artifact_id=artifact_id,
            )
            all_candidates.append(
                {**ident.to_dict(), "type": artifact_type, "similarity": similarities[i]}
            )
            if i < 5:
                relevant_artifacts.setdefault(artifact_type, []).append(ident)

        # Create gold query with candidates (manual review needed)
        gold_query = GoldQuery(
            id=query_id,
            category=category,
            query=query_text,
            language=language,
            relevant_artifacts=relevant_artifacts,
            difficulty=difficulty,
            notes=f"Auto-generated from top {top_k} candidates. Requires manual review.",
            metadata={
                "original_expected": query_data.get("expected_artifacts", []),
                "all_candidates": all_candidates,
                "candidate_scores": similarities,
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
