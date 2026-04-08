"""Orchestration script for indexing metadata from configured sources."""

import argparse
import logging
import sys
from pathlib import Path
from urllib.parse import quote_plus

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.embeddings.service import EmbeddingService
from src.indexing.pipeline import IndexingPipeline
from src.vectorstore.chroma_store import ChromaVectorStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main orchestration function."""
    parser = argparse.ArgumentParser(
        description="Index metadata from databases and code sources"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config file (default: use CONFIG_PROFILE env var)",
    )
    parser.add_argument(
        "--source",
        type=str,
        action="append",
        help="Additional source path or database URL (can be specified multiple times)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset vector store before indexing",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        default=True,
        help="Validate artifacts before indexing (default: True)",
    )

    args = parser.parse_args()

    # Load configuration
    logger.info("Loading configuration...")
    config = load_config(args.config)

    logger.info("Configuration loaded:")
    logger.info(f"  Embedding model: {config.embedding.model_name}")
    logger.info(f"  Vector store: {config.vectorstore.collection_name}")
    logger.info(f"  LLM provider: {config.llm.provider}")

    # Initialize services
    logger.info("Initializing services...")

    embedding_service = EmbeddingService(
        model_name=config.embedding.model_name,
        device=config.embedding.device,
        cache_dir=config.embedding.cache_dir,
        batch_size=config.embedding.batch_size,
        local_files_only=config.embedding.local_files_only,
    )

    vector_store = ChromaVectorStore(
        persist_directory=config.vectorstore.persist_directory,
        collection_name=config.vectorstore.collection_name,
        distance_metric=config.vectorstore.distance_metric,
    )

    # Reset if requested
    if args.reset:
        logger.warning("Resetting vector store...")
        vector_store.delete_all()
        logger.info("Vector store reset complete")

    # Initialize pipeline
    pipeline = IndexingPipeline(
        config=config,
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    # Register extractors
    from src.extractors.java_extractor_jar import JavaExtractorJar
    pipeline.register_extractor("java", JavaExtractorJar())

    # Build source list
    sources = config.indexing.source_paths.copy()
    if args.source:
        sources.extend(args.source)

    databases = {
        name: db for name, db in config.indexing.databases.items() if db.enabled
    }

    if not sources and not databases:
        logger.warning("No sources configured. Add sources to config.indexing.source_paths or databases to config.indexing.databases")
        logger.info("Example sources:")
        logger.info("  - Code: /path/to/java/project")
        logger.info("  - Database: set enabled: true in config.indexing.databases")
        sys.exit(1)

    # Run indexing for code sources
    logger.info(f"Starting indexing of {len(sources)} code source(s) and {len(databases)} database(s)...")

    overall_stats = pipeline.discover_and_index(
        source_paths=sources,
        _extractors_config=config.indexing.extractors,
    )

    # Run indexing for configured databases
    for db_name, db_config in databases.items():
        logger.info(f"Processing database: {db_name} ({db_config.type})")
        try:
            conn = db_config.connection
            if db_config.type == "postgresql":
                url = f"postgresql://{quote_plus(conn.user)}:{quote_plus(conn.password)}@{conn.host}:{conn.port}/{conn.database}"
            elif db_config.type == "sqlite" and conn.path:
                url = f"sqlite:///{conn.path}"
            else:
                logger.warning(f"Unsupported database type '{db_config.type}' for {db_name}, skipping")
                continue

            artifacts = pipeline.extract_from_database(
                url,
                schema=db_config.schema_name,
                include_tables=db_config.include_tables or None,
            )

            if db_config.tags:
                for artifact in artifacts:
                    artifact.tags.extend(db_config.tags)

            db_stats = pipeline.run_pipeline(artifacts, validate=args.validate)
            overall_stats["sources_processed"] += 1  # type: ignore[reportOperatorIssue]
            overall_stats["total_artifacts"] += db_stats.get("total_indexed", 0)  # type: ignore[reportOperatorIssue]
            overall_stats["errors"].extend(db_stats.get("errors", []))  # type: ignore[reportAttributeAccessIssue]

        except Exception as e:
            error_msg = f"Failed to process database {db_name}: {e}"
            logger.error(error_msg)
            overall_stats["errors"].append(error_msg)  # type: ignore[reportAttributeAccessIssue]

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("INDEXING SUMMARY")
    logger.info("="*60)
    logger.info(f"Sources processed: {overall_stats['sources_processed']}")
    logger.info(f"Total artifacts indexed: {overall_stats['total_artifacts']}")
    logger.info(f"Vector store total count: {vector_store.count()}")
    logger.info(f"Errors encountered: {len(overall_stats['errors'])}")

    if overall_stats['errors']:
        logger.warning("\nErrors:")
        for error in overall_stats['errors'][:10]:  # Show first 10
            logger.warning(f"  - {error}")
        if len(overall_stats['errors']) > 10:
            logger.warning(f"  ... and {len(overall_stats['errors']) - 10} more")

    logger.info("="*60)

    # Exit with error code if there were failures
    if overall_stats['errors'] and overall_stats['total_artifacts'] == 0:
        logger.error("Indexing failed - no artifacts were indexed")
        sys.exit(1)

    logger.info("Indexing completed successfully!")


if __name__ == "__main__":
    main()
