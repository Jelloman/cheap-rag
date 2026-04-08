"""Variant definitions for A/B testing.

This module defines different configurations (variants) to compare,
such as different embedding models, chunk sizes, or retrieval parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.embeddings.service import EmbeddingService


@dataclass
class VariantConfig:
    """Configuration for an A/B test variant.

    Attributes:
        name: Variant name (e.g., "baseline", "bge-large")
        embedding_model: Model name for embeddings
        embedding_dimension: Expected embedding dimension
        top_k: Number of results to retrieve
        similarity_threshold: Minimum similarity threshold
        metadata: Additional variant configuration
    """

    name: str
    embedding_model: str
    embedding_dimension: int
    top_k: int = 5
    similarity_threshold: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.embedding_dimension,
            "top_k": self.top_k,
            "similarity_threshold": self.similarity_threshold,
            "metadata": self.metadata,
        }


class EmbeddingVariant:
    """A/B test variant with its own embedding service and vector store.

    Each variant has:
    - Its own embedding model
    - Separate vector store collection
    - Independent configuration
    """

    def __init__(
        self,
        config: VariantConfig,
        vector_store_path: str | Path,
        collection_suffix: str | None = None,
    ):
        """Initialize variant.

        Args:
            config: Variant configuration
            vector_store_path: Path to vector store directory
            collection_suffix: Optional suffix for collection name
        """
        self.config = config
        self.vector_store_path = Path(vector_store_path)
        self.collection_name = f"cheap_rag_{config.name}"
        if collection_suffix:
            self.collection_name += f"_{collection_suffix}"

        # Store base config and variant-specific settings
        from src.config import load_config

        self._base_config = load_config()
        self._embedding_model = config.embedding_model

        self.embedding_service: EmbeddingService | None = None
        self.vector_store: Any = None

    def initialize(self) -> None:
        """Initialize embedding service and vector store."""
        from src.vectorstore.chroma_store import ChromaVectorStore

        # Initialize embedding service with variant's model
        self.embedding_service = EmbeddingService(
            model_name=self._embedding_model,
            device=self._base_config.embedding.device,
            cache_dir=self._base_config.embedding.cache_dir,
            batch_size=self._base_config.embedding.batch_size,
            local_files_only=self._base_config.embedding.local_files_only,
        )

        # Initialize vector store with variant-specific collection
        self.vector_store = ChromaVectorStore(
            persist_directory=str(self.vector_store_path),
            collection_name=self.collection_name,
            distance_metric=self._base_config.vectorstore.distance_metric,
        )

    def index_artifacts(self, artifacts: list[Any]) -> None:
        """Index artifacts with this variant's embeddings.

        Args:
            artifacts: List of MetadataArtifact objects
        """
        if not self.embedding_service or not self.vector_store:
            raise RuntimeError("Variant not initialized. Call initialize() first.")

        # Generate embeddings
        texts = [artifact.to_embedding_text() for artifact in artifacts]
        embeddings = self.embedding_service.embed_batch(texts)

        # Index in vector store
        self.vector_store.index_batch(artifacts, embeddings)

    def search(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search using this variant.

        Args:
            query: Query text
            filters: Optional metadata filters

        Returns:
            List of search results with artifacts and similarities
        """
        if not self.embedding_service or not self.vector_store:
            raise RuntimeError("Variant not initialized. Call initialize() first.")

        # Embed query
        query_embedding = self.embedding_service.embed_query(query)

        # Search
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=self.config.top_k,
            filters=filters or {},
        )

        # Filter by similarity threshold
        if self.config.similarity_threshold > 0:
            results = [r for r in results if r["similarity"] >= self.config.similarity_threshold]

        return results

    def clear_index(self) -> None:
        """Clear this variant's vector store."""
        if self.vector_store:
            self.vector_store.clear()


# Predefined variant configurations
BASELINE_VARIANT = VariantConfig(
    name="baseline",
    embedding_model="sentence-transformers/all-mpnet-base-v2",
    embedding_dimension=768,
    top_k=5,
    metadata={"description": "Baseline model (current production)"},
)

BGE_LARGE_VARIANT = VariantConfig(
    name="bge-large",
    embedding_model="BAAI/bge-large-en-v1.5",
    embedding_dimension=1024,
    top_k=5,
    metadata={"description": "BGE Large model (high quality)"},
)

BGE_SMALL_VARIANT = VariantConfig(
    name="bge-small",
    embedding_model="BAAI/bge-small-en-v1.5",
    embedding_dimension=384,
    top_k=5,
    metadata={"description": "BGE Small model (fast, lower quality)"},
)

E5_LARGE_VARIANT = VariantConfig(
    name="e5-large",
    embedding_model="intfloat/e5-large-v2",
    embedding_dimension=1024,
    top_k=5,
    metadata={"description": "E5 Large model"},
)
