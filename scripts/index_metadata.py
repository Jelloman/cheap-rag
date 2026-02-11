"""Orchestration script for indexing metadata from configured sources."""

import argparse
import logging
import sys
from pathlib import Path

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

    logger.info(f"Configuration loaded:")
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
    from src.extractors.java_extractor import JavaExtractor
    pipeline.register_extractor("java", JavaExtractor())

    # Build source list
    sources = config.indexing.source_paths.copy()
    if args.source:
        sources.extend(args.source)

    if not sources:
        logger.warning("No sources configured. Add sources to config.indexing.source_paths")
        logger.info("Example sources:")
        logger.info("  - Database: postgresql://user:pass@localhost/dbname")
        logger.info("  - Code: /path/to/java/project")
        sys.exit(1)

    # Run indexing
    logger.info(f"Starting indexing of {len(sources)} sources...")

    overall_stats = pipeline.discover_and_index(
        source_paths=sources,
        extractors_config=config.indexing.extractors,
    )

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
