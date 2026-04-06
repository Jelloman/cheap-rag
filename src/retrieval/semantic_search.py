"""Semantic search over metadata artifacts using embeddings and vector similarity."""

from __future__ import annotations

import time
from dataclasses import dataclass

from src.embeddings.service import EmbeddingService
from src.extractors.base import MetadataArtifact
from src.observability import (
    StructuredLogger,
    get_correlation_id,
    record_error,
    record_operation,
    trace_function,
)
from src.retrieval.filters import MetadataFilter
from src.vectorstore.chroma_store import ChromaVectorStore

logger = StructuredLogger("cheap_rag.retrieval")


@dataclass
class SearchResult:
    """Single search result with artifact and relevance score."""

    artifact: MetadataArtifact
    similarity: float
    distance: float
    rank: int

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            "artifact": self.artifact.to_dict(),
            "similarity": self.similarity,
            "distance": self.distance,
            "rank": self.rank,
        }


@dataclass
class SearchResults:
    """Collection of search results with query metadata."""

    query: str
    results: list[SearchResult]
    total_results: int
    top_k: int
    similarity_threshold: float
    filters: MetadataFilter | None = None

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            "query": self.query,
            "results": [r.to_dict() for r in self.results],
            "total_results": self.total_results,
            "top_k": self.top_k,
            "similarity_threshold": self.similarity_threshold,
            "filters": self.filters.to_dict() if self.filters else None,
        }

    def get_artifacts(self) -> list[MetadataArtifact]:
        """Extract just the artifacts from results.

        Returns:
            List of MetadataArtifact objects.
        """
        return [r.artifact for r in self.results]

    def get_top(self, n: int) -> list[SearchResult]:
        """Get top N results.

        Args:
            n: Number of results to return.

        Returns:
            Top N search results.
        """
        return self.results[:n]


class SemanticSearch:
    """Semantic search engine for metadata artifacts.

    Combines embedding generation and vector similarity search to find
    relevant metadata artifacts based on natural language queries.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: ChromaVectorStore,
        default_top_k: int = 5,
        default_similarity_threshold: float = 0.3,
    ):
        """Initialize semantic search engine.

        Args:
            embedding_service: Service for generating query embeddings.
            vector_store: Vector store for similarity search.
            default_top_k: Default number of results to return.
            default_similarity_threshold: Default minimum similarity score (0-1).
        """
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.default_top_k = default_top_k
        self.default_similarity_threshold = default_similarity_threshold

        logger.info(
            "SemanticSearch initialized",
            default_top_k=default_top_k,
            default_similarity_threshold=default_similarity_threshold,
        )

    @trace_function("semantic_search")
    def search(
        self,
        query: str,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
        filters: MetadataFilter | None = None,
    ) -> SearchResults:
        """Search for relevant metadata artifacts using semantic similarity.

        Args:
            query: Natural language query text.
            top_k: Number of results to return (uses default if None).
            similarity_threshold: Minimum similarity score 0-1 (uses default if None).
            filters: Optional metadata filters to apply.

        Returns:
            SearchResults object containing ranked artifacts.
        """
        start = time.perf_counter()

        # Use defaults if not specified
        if top_k is None:
            top_k = self.default_top_k
        if similarity_threshold is None:
            similarity_threshold = self.default_similarity_threshold

        logger.info(
            "Searching",
            query=query,
            top_k=top_k,
            threshold=similarity_threshold,
        )
        if filters and not filters.is_empty():
            logger.debug("Applying filters", filters=str(filters.to_dict()))

        try:
            # Generate query embedding
            query_embedding = self.embedding_service.embed_text(query)

            # Convert filters to ChromaDB format
            filter_dict = filters.to_dict() if filters else None

            # Search vector store
            ids, metadatas, distances = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k,
                filters=filter_dict,
            )

            # Convert distances to similarities (cosine distance -> similarity)
            # ChromaDB returns cosine distance (0 = identical, 2 = opposite)
            # Convert to similarity: 1 - (distance / 2) gives range [0, 1]
            similarities = [1.0 - (d / 2.0) for d in distances]

            # Filter by similarity threshold
            results = []
            for i, (artifact_id, metadata, distance, similarity) in enumerate(
                zip(ids, metadatas, distances, similarities, strict=True)
            ):
                if similarity >= similarity_threshold:
                    # Reconstruct MetadataArtifact from metadata
                    artifact = self._metadata_to_artifact(artifact_id, metadata)  # type: ignore[reportUnknownMemberType]
                    result = SearchResult(
                        artifact=artifact,
                        similarity=similarity,
                        distance=distance,
                        rank=i + 1,
                    )
                    results.append(result)  # type: ignore[reportUnknownMemberType]

            duration_ms = (time.perf_counter() - start) * 1000
            record_operation(
                "semantic_search",
                duration_ms,
                {"query_length": len(query), "num_results": len(results), "top_k": top_k},  # type: ignore[reportUnknownArgumentType]
            )

            logger.info(
                "Search complete",
                num_results=len(results),  # type: ignore[reportUnknownArgumentType]
                threshold=similarity_threshold,
                duration_ms=round(duration_ms, 1),
            )
            if results:
                logger.info(
                    "Top result",
                    name=results[0].artifact.name,  # type: ignore[reportUnknownMemberType]
                    similarity=f"{results[0].similarity:.3f}",  # type: ignore[reportUnknownMemberType]
                )

            return SearchResults(
                query=query,
                results=results,  # type: ignore[reportUnknownArgumentType]
                total_results=len(results),  # type: ignore[reportUnknownArgumentType]
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                filters=filters,
            )
        except Exception as e:
            record_error("retrieval", e, get_correlation_id())
            raise

    def search_similar(
        self,
        artifact: MetadataArtifact,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
        filters: MetadataFilter | None = None,
    ) -> SearchResults:
        """Find artifacts similar to a given artifact.

        Args:
            artifact: Reference artifact to find similar items.
            top_k: Number of results to return.
            similarity_threshold: Minimum similarity score.
            filters: Optional metadata filters.

        Returns:
            SearchResults object.
        """
        # Use artifact's embedding text as query
        query_text = artifact.to_embedding_text()
        return self.search(
            query=query_text,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            filters=filters,
        )

    def _metadata_to_artifact(self, artifact_id: str, metadata: dict) -> MetadataArtifact:  # type: ignore[reportMissingTypeArgument,reportUnknownParameterType]
        """Reconstruct MetadataArtifact from ChromaDB metadata.

        Args:
            artifact_id: Artifact ID.
            metadata: Metadata dictionary from ChromaDB.

        Returns:
            Reconstructed MetadataArtifact.
        """
        # Extract tags from comma-separated string
        tags = []
        if "tags" in metadata and metadata["tags"]:
            tags = metadata["tags"].split(",")  # type: ignore[reportUnknownMemberType,reportUnknownVariableType]

        # Build custom metadata dict
        custom_metadata = {}
        for key in [
            "table_name",
            "column_type",
            "nullable",
            "primary_key",
            "foreign_key",
            "unique",
            "indexed",
            "cardinality",
            "from_table",
            "to_table",
            "from_columns",
            "to_columns",
        ]:
            if key in metadata:
                value = metadata[key]  # type: ignore[reportUnknownVariableType]
                # Convert string "True"/"False" to boolean
                if value == "True":
                    value = True
                elif value == "False":
                    value = False
                custom_metadata[key] = value

        # Note: We don't have constraints, relations, examples in ChromaDB metadata
        # These would need to be stored if needed for search results
        # For now, return minimal artifact reconstruction

        return MetadataArtifact(
            id=artifact_id,
            name=metadata.get("name", ""),  # type: ignore[reportUnknownMemberType,reportUnknownArgumentType]
            type=metadata.get("type", ""),  # type: ignore[reportUnknownMemberType,reportUnknownArgumentType]
            source_type=metadata.get("source_type", ""),  # type: ignore[reportUnknownMemberType,reportUnknownArgumentType]
            language=metadata.get("language", ""),  # type: ignore[reportUnknownMemberType,reportUnknownArgumentType]
            module=metadata.get("module", ""),  # type: ignore[reportUnknownMemberType,reportUnknownArgumentType]
            description=metadata.get("description", ""),  # type: ignore[reportUnknownMemberType,reportUnknownArgumentType]
            tags=tags,  # type: ignore[reportUnknownArgumentType]
            source_file=metadata.get("source_file", ""),  # type: ignore[reportUnknownMemberType,reportUnknownArgumentType]
            source_line=metadata.get("source_line", 0),  # type: ignore[reportUnknownMemberType,reportUnknownArgumentType]
            constraints=[],  # Not stored in ChromaDB metadata
            relations=[],  # Not stored in ChromaDB metadata
            examples=[],  # Not stored in ChromaDB metadata
            metadata=custom_metadata,  # type: ignore[reportUnknownArgumentType]
        )
