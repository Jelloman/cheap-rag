"""Vector store protocols and implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from src.vectorstore.chroma_store import ChromaVectorStore

if TYPE_CHECKING:
    from src.extractors.base import MetadataArtifact


@runtime_checkable
class VectorStore(Protocol):
    """Protocol for vector stores supporting artifact storage and retrieval."""

    def add_artifacts(
        self,
        artifacts: list[MetadataArtifact],
        embeddings: np.ndarray,
    ) -> None:
        """Add artifacts with embeddings.

        Args:
            artifacts: List of metadata artifacts to store.
            embeddings: Numpy array of embeddings (shape: [n_artifacts, embedding_dim]).
        """
        ...

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[str], list[dict[str, Any]], list[float]]:
        """Search for similar artifacts.

        Args:
            query_embedding: Query embedding vector.
            top_k: Number of results to return.
            filters: Optional metadata filters.

        Returns:
            Tuple of (artifact_ids, metadata_dicts, distances).
        """
        ...

    def count(self) -> int:
        """Get number of artifacts in the store.

        Returns:
            Total count of stored artifacts.
        """
        ...


__all__ = ["VectorStore", "ChromaVectorStore"]
