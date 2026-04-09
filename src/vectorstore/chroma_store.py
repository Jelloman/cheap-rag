"""ChromaDB vector store integration for metadata search."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
import numpy as np

from src.extractors.base import MetadataArtifact

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """Vector store using ChromaDB for semantic search over metadata artifacts.

    Stores embeddings with full metadata for filtering and retrieval.
    """

    def __init__(
        self,
        persist_directory: str = "./data/vector_db",
        collection_name: str = "cheap_metadata_v1",
        distance_metric: str = "cosine",
    ):
        """Initialize ChromaDB vector store.

        Args:
            persist_directory: Directory for persistent storage
            collection_name: Name of the collection
            distance_metric: Distance metric ("cosine", "l2", "ip")
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.distance_metric = distance_metric

        # Create persist directory
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initializing ChromaDB at: {self.persist_directory}")

        # Create ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
            ),
        )

        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"Loaded existing collection: {collection_name}")
            logger.info(f"  Items in collection: {self.collection.count()}")
        except Exception:
            # Collection doesn't exist, create it
            metadata_config = {
                "hnsw:space": distance_metric,
            }
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata=metadata_config,
            )
            logger.info(f"Created new collection: {collection_name}")

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

        # Prepare data for ChromaDB
        ids = [a.id for a in artifacts]
        documents = [a.to_embedding_text() for a in artifacts]
        metadatas = [self._artifact_to_metadata(a) for a in artifacts]
        embeddings_list = embeddings.tolist()

        # Add to collection in batches
        batch_size = 100
        for i in range(0, len(artifacts), batch_size):
            end_idx = min(i + batch_size, len(artifacts))

            self.collection.upsert(
                ids=ids[i:end_idx],
                documents=documents[i:end_idx],
                metadatas=metadatas[i:end_idx],  # type: ignore[reportArgumentType]  # chromadb
                embeddings=embeddings_list[i:end_idx],
            )

            logger.info(
                f"  Added batch {i // batch_size + 1} ({i + 1}-{end_idx} of {len(artifacts)})"
            )

        logger.info(f"Successfully added {len(artifacts)} artifacts")
        logger.info(f"Total items in collection: {self.collection.count()}")

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
        # Convert numpy array to list
        if isinstance(query_embedding, np.ndarray):  # type: ignore[reportUnnecessaryIsInstance]  # chromadb
            query_embedding = query_embedding.tolist()

        # Build where clause for filtering
        where_clause = None
        if filters:
            where_clause = self._build_where_clause(filters)

        # Query the collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_clause,
            include=["metadatas", "distances", "documents"],
        )

        # Extract results
        ids = results["ids"][0] if results["ids"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []
        distances = results["distances"][0] if results["distances"] else []

        return ids, metadatas, distances  # type: ignore[reportReturnType]  # chromadb

    def get_artifact(self, artifact_id: str) -> dict[str, Any] | None:
        """Retrieve a specific artifact by ID.

        Args:
            artifact_id: Artifact ID

        Returns:
            Artifact metadata dict or None if not found
        """
        try:
            results = self.collection.get(
                ids=[artifact_id],
                include=["metadatas", "documents"],
            )

            if results["ids"]:
                return results["metadatas"][0]  # type: ignore[reportReturnType,reportUnknownVariableType]  # chromadb
            return None
        except Exception as e:
            logger.error(f"Error retrieving artifact {artifact_id}: {e}")
            return None

    def delete_all(self) -> None:
        """Delete all artifacts from the collection."""
        logger.warning(f"Deleting all data from collection: {self.collection_name}")
        self.client.delete_collection(name=self.collection_name)

        # Recreate empty collection
        metadata_config = {
            "hnsw:space": self.distance_metric,
        }
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata=metadata_config,
        )
        logger.info("Collection recreated (empty)")

    def count(self) -> int:
        """Get the number of artifacts in the collection.

        Returns:
            Number of artifacts
        """
        return self.collection.count()

    def get_all_documents(self) -> tuple[list[str], list[str], list[dict[str, Any]]]:
        """Retrieve all documents from the collection.

        Returns:
            Tuple of (ids, documents, metadatas)
        """
        results = self.collection.get(include=["documents", "metadatas"])
        ids: list[str] = results["ids"] or []
        documents: list[str] = results["documents"] or []  # type: ignore[reportAssignmentType]  # chromadb
        metadatas: list[dict[str, Any]] = results["metadatas"] or []  # type: ignore[reportAssignmentType]  # chromadb
        return ids, documents, metadatas

    def add_raw(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Add pre-embedded documents directly to the collection.

        Args:
            ids: Artifact IDs
            embeddings: Pre-computed embedding vectors
            documents: Document texts
            metadatas: Metadata dicts
        """
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            end_idx = min(i + batch_size, len(ids))
            self.collection.upsert(
                ids=ids[i:end_idx],
                embeddings=embeddings[i:end_idx],
                documents=documents[i:end_idx],
                metadatas=metadatas[i:end_idx],  # type: ignore[reportArgumentType]  # chromadb
            )
            logger.info(f"  Added batch {i // batch_size + 1} ({i + 1}-{end_idx} of {len(ids)})")
        logger.info(f"Total items in collection: {self.collection.count()}")

    def _artifact_to_metadata(self, artifact: MetadataArtifact) -> dict[str, Any]:
        """Convert artifact to ChromaDB metadata dict.

        ChromaDB metadata must be JSON-serializable with limited types.
        """
        metadata = {
            "name": artifact.name,
            "type": artifact.type,
            "source_type": artifact.source_type,
            "language": artifact.language,
            "module": artifact.module,
            "description": artifact.description[:1000],  # Limit length
            "source_file": artifact.source_file,
            "source_line": artifact.source_line,
        }

        # Add tags as comma-separated string (ChromaDB doesn't support lists in filters)
        if artifact.tags:
            metadata["tags"] = ",".join(artifact.tags)

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
                value = artifact.metadata[key]
                # Convert to JSON-serializable type
                if isinstance(value, bool):
                    metadata[key] = value
                elif value is not None:
                    metadata[key] = str(value)

        # Compute and store component grouping key so it can be used as a WHERE filter
        # during gold dataset ID resolution (see find_artifact_id).
        lang = artifact.language
        atype = artifact.type
        if lang == "java" and artifact.source_file:
            metadata["component"] = Path(artifact.source_file).stem
        elif lang == "postgresql":
            if atype in {"column", "index", "constraint", "trigger"}:
                metadata["component"] = str(artifact.metadata.get("table_name", ""))
            elif atype == "relationship":
                metadata["component"] = str(artifact.metadata.get("from_table", ""))
            else:
                metadata["component"] = artifact.name

        return metadata

    def find_artifact_id(
        self,
        name: str,
        artifact_type: str,
        language: str,
        component: str | None = None,
    ) -> str | None:
        """Look up a single artifact ID by human-readable identifiers.

        Intended for resolving gold dataset entries that were hand-authored without
        an explicit ``artifact_id``.

        For PostgreSQL child types (column, index, constraint, trigger) ``component``
        maps to the ``table_name`` ChromaDB field; for relationships it maps to
        ``from_table``.  For Java members ``component`` maps to the ``component``
        field (stored as the source-file stem).  Top-level artifacts need only
        ``name`` and ``artifact_type``.

        Note: requires that the index was built with a version of the pipeline that
        stores the ``component`` field (i.e. re-indexing is needed for older indexes).

        Args:
            name: Artifact name (e.g. "film", "customer_id", "Catalog").
            artifact_type: ChromaDB type string (e.g. "table", "column", "class").
            language: Source language (e.g. "postgresql", "java").
            component: Owning table (Postgres) or class name (Java) for child artifacts.

        Returns:
            Artifact ID string, or None if not found or ambiguous.
        """
        conditions: list[dict[str, Any]] = [
            {"name": name},
            {"type": artifact_type},
            {"language": language},
        ]

        if component:
            if language == "postgresql":
                if artifact_type in {"column", "index", "constraint", "trigger"}:
                    conditions.append({"table_name": component})
                elif artifact_type == "relationship":
                    conditions.append({"from_table": component})
                # for top-level postgres types component == name, no extra filter needed
            else:
                # Java and others: component is stored directly
                conditions.append({"component": component})

        where: dict[str, Any] = {"$and": conditions} if len(conditions) > 1 else conditions[0]

        try:
            results = self.collection.get(where=where, include=["metadatas"])
            ids: list[str] = results["ids"]  # type: ignore[assignment]
            if not ids:
                return None
            if len(ids) > 1:
                logger.warning(
                    "Ambiguous artifact lookup: %s/%s/%s%s matched %d artifacts — using first",
                    language,
                    artifact_type,
                    name,
                    f"/{component}" if component else "",
                    len(ids),
                )
            return ids[0]
        except Exception as exc:
            logger.error(
                "Error looking up artifact %s/%s/%s: %s", language, artifact_type, name, exc
            )
            return None

    def _build_where_clause(self, filters: dict[str, Any]) -> dict[str, Any]:
        """Build ChromaDB where clause from filter dict.

        Args:
            filters: Filter dict (e.g., {"language": "java", "type": "class"})

        Returns:
            ChromaDB where clause
        """
        conditions = []

        for key, value in filters.items():
            if isinstance(value, list):
                # Multiple values: use $in operator
                conditions.append({key: {"$in": value}})  # type: ignore[reportUnknownMemberType]  # chromadb
            else:
                # Single value: direct equality
                conditions.append({key: value})  # type: ignore[reportUnknownMemberType]  # chromadb

        # Combine with $and if multiple conditions
        if len(conditions) == 1:  # type: ignore[reportUnknownArgumentType]  # chromadb
            return conditions[0]  # type: ignore[reportUnknownVariableType]  # chromadb
        elif len(conditions) > 1:  # type: ignore[reportUnknownArgumentType]  # chromadb
            return {"$and": conditions}
        else:
            return {}
