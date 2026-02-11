"""Unified indexing pipeline for metadata extraction and embedding."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import create_engine

from src.config import Config, ExtractorConfig
from src.embeddings.service import EmbeddingService
from src.extractors.base import MetadataArtifact, MetadataExtractor
from src.extractors.database_extractor import DatabaseExtractor
from src.extractors.postgres_extractor import PostgresExtractor
from src.extractors.sqlite_extractor import SQLiteExtractor
from src.extractors.java_extractor import JavaExtractor
from src.indexing.schema import validate_artifact
from src.vectorstore.chroma_store import ChromaVectorStore

logger = logging.getLogger(__name__)


class ExtractionError(Exception):
    """Error during metadata extraction."""
    pass


class IndexingPipeline:
    """Unified pipeline for extracting, validating, embedding, and indexing metadata."""

    def __init__(
        self,
        config: Config,
        embedding_service: EmbeddingService,
        vector_store: ChromaVectorStore,
    ):
        """Initialize indexing pipeline.

        Args:
            config: System configuration.
            embedding_service: Service for generating embeddings.
            vector_store: Vector store for indexing.
        """
        self.config = config
        self.embedding_service = embedding_service
        self.vector_store = vector_store

        # Registry of extractors
        self.extractors: Dict[str, MetadataExtractor] = {}

        # Statistics
        self.stats = {
            "total_extracted": 0,
            "total_validated": 0,
            "total_embedded": 0,
            "total_indexed": 0,
            "errors": [],
        }

        logger.info("IndexingPipeline initialized")

    def register_extractor(self, name: str, extractor: MetadataExtractor) -> None:
        """Register a metadata extractor.

        Args:
            name: Extractor name (e.g., "java", "postgres").
            extractor: Extractor instance.
        """
        self.extractors[name] = extractor
        logger.info(f"Registered extractor: {name}")

    def extract_from_source(
        self,
        source_path: Path,
        extractor_name: str,
    ) -> List[MetadataArtifact]:
        """Extract metadata from a source using specified extractor.

        Args:
            source_path: Path to source file or directory.
            extractor_name: Name of registered extractor.

        Returns:
            List of extracted artifacts.

        Raises:
            ExtractionError: If extraction fails.
        """
        if extractor_name not in self.extractors:
            raise ExtractionError(f"Extractor not found: {extractor_name}")

        extractor = self.extractors[extractor_name]

        logger.info(f"Extracting from {source_path} using {extractor_name}...")

        try:
            artifacts = extractor.extract_metadata(source_path)
            logger.info(f"Extracted {len(artifacts)} artifacts")
            return artifacts

        except Exception as e:
            error_msg = f"Extraction failed for {source_path}: {e}"
            logger.error(error_msg)
            self.stats["errors"].append(error_msg)
            raise ExtractionError(error_msg) from e

    def extract_from_database(
        self,
        db_url: str,
        extractor_name: str = "database",
        schema: Optional[str] = None,
    ) -> List[MetadataArtifact]:
        """Extract metadata from a database.

        Args:
            db_url: SQLAlchemy database URL.
            extractor_name: Database extractor name.
            schema: Optional schema name to extract.

        Returns:
            List of extracted artifacts.
        """
        logger.info(f"Extracting from database: {db_url}")

        try:
            # Create database engine
            engine = create_engine(db_url)

            # Get appropriate extractor
            if "postgresql" in db_url:
                extractor = PostgresExtractor(engine, schema_name=schema)
            elif "sqlite" in db_url:
                extractor = SQLiteExtractor(engine)
            else:
                extractor = DatabaseExtractor(engine)

            # Extract metadata
            artifacts = extractor.extract_metadata()
            logger.info(f"Extracted {len(artifacts)} artifacts from database")

            return artifacts

        except Exception as e:
            error_msg = f"Database extraction failed: {e}"
            logger.error(error_msg)
            self.stats["errors"].append(error_msg)
            raise ExtractionError(error_msg) from e

    def validate_artifacts(
        self,
        artifacts: List[MetadataArtifact],
    ) -> List[MetadataArtifact]:
        """Validate artifacts against schema.

        Args:
            artifacts: List of artifacts to validate.

        Returns:
            List of valid artifacts.
        """
        logger.info(f"Validating {len(artifacts)} artifacts...")

        valid_artifacts = []
        invalid_count = 0

        for artifact in artifacts:
            is_valid, errors = validate_artifact(artifact)
            if is_valid:
                valid_artifacts.append(artifact)
            else:
                invalid_count += 1
                error_msg = f"Invalid artifact {artifact.id}: {errors}"
                logger.warning(error_msg)
                self.stats["errors"].append(error_msg)

        logger.info(f"Validation: {len(valid_artifacts)} valid, {invalid_count} invalid")
        return valid_artifacts

    def embed_artifacts(
        self,
        artifacts: List[MetadataArtifact],
    ) -> tuple[List[MetadataArtifact], Any]:
        """Generate embeddings for artifacts.

        Args:
            artifacts: List of artifacts to embed.

        Returns:
            Tuple of (artifacts, embeddings).
        """
        if not artifacts:
            logger.warning("No artifacts to embed")
            return artifacts, None

        logger.info(f"Generating embeddings for {len(artifacts)} artifacts...")

        try:
            embeddings = self.embedding_service.embed_artifacts(artifacts)
            logger.info(f"Generated {len(embeddings)} embeddings")
            return artifacts, embeddings

        except Exception as e:
            error_msg = f"Embedding generation failed: {e}"
            logger.error(error_msg)
            self.stats["errors"].append(error_msg)
            raise

    def index_artifacts(
        self,
        artifacts: List[MetadataArtifact],
        embeddings: Any,
    ) -> None:
        """Index artifacts with embeddings in vector store.

        Args:
            artifacts: List of artifacts.
            embeddings: Embedding matrix.
        """
        if not artifacts:
            logger.warning("No artifacts to index")
            return

        logger.info(f"Indexing {len(artifacts)} artifacts...")

        try:
            self.vector_store.add_artifacts(artifacts, embeddings)
            logger.info(f"Successfully indexed {len(artifacts)} artifacts")

        except Exception as e:
            error_msg = f"Indexing failed: {e}"
            logger.error(error_msg)
            self.stats["errors"].append(error_msg)
            raise

    def run_pipeline(
        self,
        artifacts: List[MetadataArtifact],
        validate: bool = True,
    ) -> Dict[str, int]:
        """Run full pipeline on extracted artifacts.

        Args:
            artifacts: Pre-extracted artifacts.
            validate: Whether to validate artifacts.

        Returns:
            Statistics dictionary.
        """
        logger.info(f"Running pipeline on {len(artifacts)} artifacts...")

        self.stats["total_extracted"] = len(artifacts)

        # Validation
        if validate:
            artifacts = self.validate_artifacts(artifacts)
            self.stats["total_validated"] = len(artifacts)
        else:
            self.stats["total_validated"] = len(artifacts)

        # Embedding
        artifacts, embeddings = self.embed_artifacts(artifacts)
        self.stats["total_embedded"] = len(artifacts)

        # Indexing
        self.index_artifacts(artifacts, embeddings)
        self.stats["total_indexed"] = len(artifacts)

        logger.info("Pipeline completed")
        logger.info(f"  Extracted: {self.stats['total_extracted']}")
        logger.info(f"  Validated: {self.stats['total_validated']}")
        logger.info(f"  Embedded: {self.stats['total_embedded']}")
        logger.info(f"  Indexed: {self.stats['total_indexed']}")
        logger.info(f"  Errors: {len(self.stats['errors'])}")

        return self.stats

    def index_database(
        self,
        db_url: str,
        schema: Optional[str] = None,
        validate: bool = True,
    ) -> Dict[str, int]:
        """Extract and index metadata from a database.

        Args:
            db_url: Database connection URL.
            schema: Optional schema name.
            validate: Whether to validate artifacts.

        Returns:
            Statistics dictionary.
        """
        logger.info(f"Indexing database: {db_url}")

        # Extract
        artifacts = self.extract_from_database(db_url, schema=schema)

        # Run pipeline
        return self.run_pipeline(artifacts, validate=validate)

    def index_code(
        self,
        source_path: Path,
        language: str,
        validate: bool = True,
    ) -> Dict[str, int]:
        """Extract and index metadata from code.

        Args:
            source_path: Path to code files/directory.
            language: Programming language ("java", "typescript", etc.).
            validate: Whether to validate artifacts.

        Returns:
            Statistics dictionary.
        """
        logger.info(f"Indexing {language} code from: {source_path}")

        # Extract
        artifacts = self.extract_from_source(source_path, language)

        # Run pipeline
        return self.run_pipeline(artifacts, validate=validate)

    def discover_and_index(
        self,
        source_paths: List[str],
        extractors_config: Dict[str, ExtractorConfig],
    ) -> Dict[str, Any]:
        """Discover and index from multiple sources.

        Args:
            source_paths: List of source paths or database URLs.
            extractors_config: Extractor configuration.

        Returns:
            Overall statistics.
        """
        logger.info("Starting discovery and indexing...")

        overall_stats = {
            "sources_processed": 0,
            "total_artifacts": 0,
            "errors": [],
        }

        for source in source_paths:
            logger.info(f"Processing source: {source}")

            try:
                # Determine source type
                if source.startswith("postgresql://") or source.startswith("sqlite://"):
                    # Database source
                    stats = self.index_database(source)
                else:
                    # Code source - try to infer language
                    source_path = Path(source)

                    if not source_path.exists():
                        logger.warning(f"Source path does not exist: {source}")
                        continue

                    # Try Java first
                    if any(source_path.glob("**/*.java")):
                        stats = self.index_code(source_path, "java")
                    else:
                        logger.warning(f"Could not determine language for: {source}")
                        continue

                overall_stats["sources_processed"] += 1
                overall_stats["total_artifacts"] += stats.get("total_indexed", 0)
                overall_stats["errors"].extend(stats.get("errors", []))

            except Exception as e:
                error_msg = f"Failed to process {source}: {e}"
                logger.error(error_msg)
                overall_stats["errors"].append(error_msg)

        logger.info("Discovery and indexing completed")
        logger.info(f"  Sources processed: {overall_stats['sources_processed']}")
        logger.info(f"  Total artifacts indexed: {overall_stats['total_artifacts']}")
        logger.info(f"  Total errors: {len(overall_stats['errors'])}")

        return overall_stats

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics.

        Returns:
            Statistics dictionary.
        """
        return {
            **self.stats,
            "vector_store_count": self.vector_store.count(),
        }

    def reset_stats(self) -> None:
        """Reset pipeline statistics."""
        self.stats = {
            "total_extracted": 0,
            "total_validated": 0,
            "total_embedded": 0,
            "total_indexed": 0,
            "errors": [],
        }
        logger.info("Pipeline statistics reset")
