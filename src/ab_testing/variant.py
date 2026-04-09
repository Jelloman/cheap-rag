"""Variant definitions for A/B testing.

This module defines different configurations (variants) to compare,
such as different embedding models, chunk sizes, or retrieval parameters.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.embeddings.service import EmbeddingService


def _is_model_cached(model_name: str, cache_dir: str | Path | None) -> bool:
    """Return True if the model's files are already in the local HuggingFace cache.

    When True, EmbeddingService can be initialised with local_files_only=True,
    which skips the Hub version-check request entirely.
    """
    try:
        from huggingface_hub import try_to_load_from_cache  # type: ignore[reportMissingTypeStubs]

        result = try_to_load_from_cache(
            repo_id=model_name,
            filename="config.json",
            cache_dir=str(cache_dir) if cache_dir else None,
        )
        # Returns a path string when cached, None or a sentinel object when not
        return isinstance(result, str)
    except Exception:
        return False

logger = logging.getLogger(__name__)


@dataclass
class VariantConfig:
    """Configuration for an A/B test variant.

    Attributes:
        name: Variant name (e.g., "baseline_mpnet", "bge_large")
        embedding_model: HuggingFace model name for embeddings
        embedding_dimension: Expected embedding dimension
        top_k: Number of results to retrieve
        similarity_threshold: Minimum cosine similarity (0 = disabled)
        use_existing_index: If True, connect to an already-indexed collection
            instead of re-indexing. Use this for the baseline model.
        existing_persist_directory: ChromaDB path for an existing collection
            (only used when use_existing_index=True)
        existing_collection_name: Collection name for an existing index
            (only used when use_existing_index=True)
        source_persist_directory: ChromaDB path to pull documents from for
            re-indexing (only used when use_existing_index=False)
        source_collection_name: Source collection name for re-indexing
            (only used when use_existing_index=False)
        metadata: Additional variant configuration
    """

    name: str
    embedding_model: str
    embedding_dimension: int
    description: str = ""
    top_k: int = 5
    similarity_threshold: float = 0.0
    use_existing_index: bool = False
    existing_persist_directory: str = ""
    existing_collection_name: str = ""
    source_persist_directory: str = ""
    source_collection_name: str = ""
    query_prefix: str = ""
    document_prefix: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> VariantConfig:
        """Load a VariantConfig from a YAML file.

        Args:
            path: Path to the YAML config file

        Returns:
            VariantConfig instance
        """
        with open(path) as f:
            data: dict[str, Any] = yaml.safe_load(f)
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.embedding_dimension,
            "top_k": self.top_k,
            "similarity_threshold": self.similarity_threshold,
            "use_existing_index": self.use_existing_index,
            "query_prefix": self.query_prefix,
            "document_prefix": self.document_prefix,
            "metadata": self.metadata,
        }


class EmbeddingVariant:
    """A/B test variant with its own embedding service and vector store.

    Two modes:
    - use_existing_index=True: connects to an already-indexed collection (baseline).
      No re-indexing is done; the collection is used read-only for search.
    - use_existing_index=False: pulls all documents from a source collection,
      re-embeds them with this variant's model, and indexes into a new collection.
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
            vector_store_path: Base path for new variant vector stores
                (ignored when use_existing_index=True)
            collection_suffix: Optional suffix appended to the collection name
                (ignored when use_existing_index=True)
        """
        self.config = config
        self.vector_store_path = Path(vector_store_path)
        self.collection_name = f"cheap_rag_{config.name}"
        if collection_suffix:
            self.collection_name += f"_{collection_suffix}"

        from src.config import load_config

        self._base_config = load_config()
        self.embedding_service: EmbeddingService | None = None
        self.vector_store: Any = None

    def initialize(self) -> None:
        """Initialize embedding service and vector store.

        For use_existing_index variants: connects to the existing collection.
        For re-index variants: pulls documents from the source collection,
        re-embeds with this model, and writes a new collection.
        """
        from src.vectorstore.chroma_store import ChromaVectorStore

        if self.config.use_existing_index:
            self._initialize_existing()
        else:
            self._initialize_reindex()

    def _initialize_existing(self) -> None:
        """Connect to an existing indexed collection (baseline path)."""
        from src.vectorstore.chroma_store import ChromaVectorStore

        if not self.config.existing_persist_directory or not self.config.existing_collection_name:
            raise ValueError(
                f"Variant '{self.config.name}' has use_existing_index=True but "
                "existing_persist_directory / existing_collection_name are not set"
            )

        logger.info(
            f"Variant '{self.config.name}': connecting to existing collection "
            f"'{self.config.existing_collection_name}' at "
            f"'{self.config.existing_persist_directory}'"
        )

        # Baseline model is already cached; respect local_files_only from config
        self.embedding_service = EmbeddingService(
            model_name=self.config.embedding_model,
            device=self._base_config.embedding.device,
            cache_dir=self._base_config.embedding.cache_dir,
            batch_size=self._base_config.embedding.batch_size,
            local_files_only=self._base_config.embedding.local_files_only,
        )
        self.vector_store = ChromaVectorStore(
            persist_directory=self.config.existing_persist_directory,
            collection_name=self.config.existing_collection_name,
            distance_metric=self._base_config.vectorstore.distance_metric,
        )

    def _initialize_reindex(self) -> None:
        """Pull documents from source collection and re-index with this model."""
        from src.vectorstore.chroma_store import ChromaVectorStore

        if not self.config.source_persist_directory or not self.config.source_collection_name:
            raise ValueError(
                f"Variant '{self.config.name}' has use_existing_index=False but "
                "source_persist_directory / source_collection_name are not set. "
                "Set these fields in the variant YAML to specify where to pull documents from."
            )

        cached = _is_model_cached(
            self.config.embedding_model, self._base_config.embedding.cache_dir
        )
        logger.info(
            f"Variant '{self.config.name}': loading model '{self.config.embedding_model}' "
            f"({'cached' if cached else 'downloading'})"
        )

        self.embedding_service = EmbeddingService(
            model_name=self.config.embedding_model,
            device=self._base_config.embedding.device,
            cache_dir=self._base_config.embedding.cache_dir,
            batch_size=self._base_config.embedding.batch_size,
            local_files_only=cached,
        )

        # Open destination collection
        self.vector_store = ChromaVectorStore(
            persist_directory=str(self.vector_store_path),
            collection_name=self.collection_name,
            distance_metric=self._base_config.vectorstore.distance_metric,
        )

        # Skip re-indexing if the collection already has data
        if self.vector_store.count() > 0:
            logger.info(
                f"Variant '{self.config.name}': collection already has "
                f"{self.vector_store.count()} items, skipping re-indexing"
            )
            return

        # Pull documents from source collection
        logger.info(
            f"Variant '{self.config.name}': pulling documents from "
            f"'{self.config.source_collection_name}' at "
            f"'{self.config.source_persist_directory}'"
        )
        source_store = ChromaVectorStore(
            persist_directory=self.config.source_persist_directory,
            collection_name=self.config.source_collection_name,
            distance_metric=self._base_config.vectorstore.distance_metric,
        )
        ids, documents, metadatas = source_store.get_all_documents()

        if not ids:
            logger.warning(
                f"Variant '{self.config.name}': source collection is empty, nothing to index"
            )
            return

        logger.info(
            f"Variant '{self.config.name}': re-embedding {len(ids)} documents "
            f"with '{self.config.embedding_model}'"
        )
        if self.config.document_prefix:
            documents = [self.config.document_prefix + d for d in documents]
        embeddings = self.embedding_service.embed_batch(documents)
        embeddings_list = embeddings.tolist()

        self.vector_store.add_raw(ids, embeddings_list, documents, metadatas)
        logger.info(f"Variant '{self.config.name}': indexing complete")

    def search(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
    ) -> list[str]:
        """Search using this variant.

        Args:
            query: Query text
            filters: Optional metadata filters

        Returns:
            List of retrieved artifact IDs
        """
        if not self.embedding_service or not self.vector_store:
            raise RuntimeError("Variant not initialized. Call initialize() first.")

        if self.config.query_prefix:
            query = self.config.query_prefix + query
        query_embedding = self.embedding_service.embed_query(query)

        ids, _metadatas, distances = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=self.config.top_k,
            filters=filters or {},
        )

        # Filter by similarity threshold (distance = 1 - cosine_similarity)
        if self.config.similarity_threshold > 0:
            ids = [
                artifact_id
                for artifact_id, dist in zip(ids, distances)
                if (1.0 - dist) >= self.config.similarity_threshold
            ]

        return ids

    def clear_index(self) -> None:
        """Clear this variant's vector store.

        Skipped for variants that use an existing index — we never wipe
        the production collection.
        """
        if self.vector_store and not self.config.use_existing_index:
            self.vector_store.delete_all()
