"""FAISS vector store integration for metadata search.

Alternative to ChromaDB that works with Python 3.14+.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from src.extractors.base import MetadataArtifact

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """Vector store using FAISS for semantic search over metadata artifacts.

    Stores embeddings with full metadata for filtering and retrieval.
    Works with Python 3.14+ (unlike ChromaDB).
    """

    def __init__(
        self,
        persist_directory: str = "./data/vector_db",
        collection_name: str = "cheap_metadata_v1",
        distance_metric: str = "cosine",
        dimension: int = 768,
    ):
        """Initialize FAISS vector store.

        Args:
            persist_directory: Directory for persistent storage
            collection_name: Name of the collection
            distance_metric: Distance metric ("cosine", "l2")
            dimension: Embedding dimension
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.distance_metric = distance_metric
        self.dimension = dimension

        # Create persist directory
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # File paths
        self.index_path = self.persist_directory / f"{collection_name}.index"
        self.metadata_path = self.persist_directory / f"{collection_name}.metadata.pkl"

        logger.info(f"Initializing FAISS at: {self.persist_directory}")

        # Load or create index
        self._load_or_create_index()

    def _load_or_create_index(self):
        """Load existing index or create new one."""
        if self.index_path.exists() and self.metadata_path.exists():
            # Load existing index
            logger.info(f"Loading existing index from: {self.index_path}")
            self.index = faiss.read_index(str(self.index_path))

            with open(self.metadata_path, "rb") as f:
                data = pickle.load(f)
                self.ids = data["ids"]
                self.metadatas = data["metadatas"]
                self.documents = data["documents"]

            logger.info(f"Loaded {len(self.ids)} items from index")
        else:
            # Create new index
            logger.info("Creating new FAISS index")

            if self.distance_metric == "cosine":
                # Normalize vectors for cosine similarity using L2 index
                self.index = faiss.IndexFlatIP(
                    self.dimension
                )  # Inner product after normalization = cosine
            else:  # l2
                self.index = faiss.IndexFlatL2(self.dimension)

            self.ids = []
            self.metadatas = []
            self.documents = []

            logger.info(f"Created new index with dimension: {self.dimension}")

    def add_artifacts(
        self,
        artifacts: list[MetadataArtifact],
        embeddings: np.ndarray,
    ) -> None:
        """Add artifacts with their embeddings to the vector store.

        Args:
            artifacts: List of metadata artifacts
            embeddings: Embedding matrix (num_artifacts x embedding_dim)
        """
        if len(artifacts) != len(embeddings):
            raise ValueError(
                f"Number of artifacts ({len(artifacts)}) must match "
                f"number of embeddings ({len(embeddings)})"
            )

        if not artifacts:
            logger.warning("No artifacts to add")
            return

        logger.info(f"Adding {len(artifacts)} artifacts to vector store...")

        # Normalize embeddings if using cosine similarity
        if self.distance_metric == "cosine":
            faiss.normalize_L2(embeddings)

        # Add to index
        self.index.add(embeddings.astype(np.float32))

        # Store metadata
        for artifact in artifacts:
            self.ids.append(artifact.id)
            self.metadatas.append(self._artifact_to_metadata(artifact))
            self.documents.append(artifact.to_embedding_text())

        logger.info(f"Successfully added {len(artifacts)} artifacts")
        logger.info(f"Total items in collection: {self.count()}")

        # Save to disk
        self._save_index()

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[str], list[dict[str, Any]], list[float]]:
        """Search for similar artifacts using a query embedding.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filters: Metadata filters (e.g., {"language": "java", "type": "class"})

        Returns:
            Tuple of (ids, metadatas, distances)
        """
        if self.count() == 0:
            logger.warning("Index is empty")
            return [], [], []

        # Ensure query embedding is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Normalize if using cosine similarity
        if self.distance_metric == "cosine":
            faiss.normalize_L2(query_embedding)

        # Search
        # Get more results than needed for filtering
        search_k = min(top_k * 10 if filters else top_k, self.count())
        distances, indices = self.index.search(query_embedding.astype(np.float32), search_k)

        # Convert to lists
        distances = distances[0].tolist()
        indices = indices[0].tolist()

        # Filter results
        result_ids = []
        result_metadatas = []
        result_distances = []

        for idx, distance in zip(indices, distances, strict=True):
            if idx < 0 or idx >= len(self.ids):
                continue

            metadata = self.metadatas[idx]

            # Apply filters
            if filters and not self._matches_filters(metadata, filters):
                continue

            result_ids.append(self.ids[idx])
            result_metadatas.append(metadata)
            result_distances.append(distance)

            if len(result_ids) >= top_k:
                break

        return result_ids, result_metadatas, result_distances

    def get_artifact(self, artifact_id: str) -> dict[str, Any] | None:
        """Retrieve a specific artifact by ID.

        Args:
            artifact_id: Artifact ID

        Returns:
            Artifact metadata dict or None if not found
        """
        try:
            idx = self.ids.index(artifact_id)
            return self.metadatas[idx]
        except ValueError:
            logger.error(f"Artifact {artifact_id} not found")
            return None

    def delete_all(self) -> None:
        """Delete all artifacts from the collection."""
        logger.warning(f"Deleting all data from collection: {self.collection_name}")

        # Recreate empty index
        if self.distance_metric == "cosine":
            self.index = faiss.IndexFlatIP(self.dimension)
        else:
            self.index = faiss.IndexFlatL2(self.dimension)

        self.ids = []
        self.metadatas = []
        self.documents = []

        # Delete persisted files
        if self.index_path.exists():
            self.index_path.unlink()
        if self.metadata_path.exists():
            self.metadata_path.unlink()

        logger.info("Collection recreated (empty)")

    def count(self) -> int:
        """Get the number of artifacts in the collection.

        Returns:
            Number of artifacts
        """
        return self.index.ntotal

    def _save_index(self):
        """Save index and metadata to disk."""
        faiss.write_index(self.index, str(self.index_path))

        with open(self.metadata_path, "wb") as f:
            pickle.dump(
                {
                    "ids": self.ids,
                    "metadatas": self.metadatas,
                    "documents": self.documents,
                },
                f,
            )

        logger.info(f"Index saved to: {self.index_path}")

    def _artifact_to_metadata(self, artifact: MetadataArtifact) -> dict[str, Any]:
        """Convert artifact to metadata dict."""
        metadata = {
            "name": artifact.name,
            "type": artifact.type,
            "source_type": artifact.source_type,
            "language": artifact.language,
            "module": artifact.module,
            "description": artifact.description,
            "source_file": artifact.source_file,
            "source_line": artifact.source_line,
            "tags": artifact.tags,
        }

        # Add selected custom metadata fields
        for key in [
            "table_name",
            "column_type",
            "nullable",
            "primary_key",
            "foreign_key",
            "cardinality",
            "from_table",
            "to_table",
        ]:
            if key in artifact.metadata:
                metadata[key] = artifact.metadata[key]

        return metadata

    def _matches_filters(self, metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
        """Check if metadata matches filter criteria."""
        for key, value in filters.items():
            if key not in metadata:
                return False

            if isinstance(value, list):
                # Check if metadata value is in list
                if metadata[key] not in value:
                    return False
            else:
                # Direct equality
                if metadata[key] != value:
                    return False

        return True
