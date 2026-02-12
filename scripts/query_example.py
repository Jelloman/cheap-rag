"""Example script demonstrating the full RAG query pipeline."""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config, get_anthropic_api_key
from src.embeddings.service import EmbeddingService
from src.generation.citations import CitationExtractor
from src.generation.generator import Generator, OllamaProvider, AnthropicProvider
from src.retrieval.filters import FilterBuilder
from src.retrieval.semantic_search import SemanticSearch
from src.vectorstore.chroma_store import ChromaVectorStore
from src.generation.response import (
    QueryResponse, SearchMetadata, GenerationMetadata,
    CitationMetrics, ArtifactSummary, CitationInfo
)
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main query function."""
    parser = argparse.ArgumentParser(description="Query metadata using RAG")
    parser.add_argument(
        "query",
        type=str,
        help="Question to ask",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config file",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to retrieve",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Similarity threshold (0-1)",
    )
    parser.add_argument(
        "--language",
        type=str,
        help="Filter by language",
    )
    parser.add_argument(
        "--type",
        type=str,
        help="Filter by artifact type",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Output as markdown",
    )

    args = parser.parse_args()

    # Load configuration
    logger.info("Loading configuration...")
    config = load_config(args.config)

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

    semantic_search = SemanticSearch(
        embedding_service=embedding_service,
        vector_store=vector_store,
        default_top_k=config.retrieval.top_k,
        default_similarity_threshold=config.retrieval.similarity_threshold,
    )

    # Initialize LLM generator
    logger.info(f"Initializing LLM provider: {config.llm.provider}")

    if config.llm.provider == "ollama":
        provider = OllamaProvider(config.llm.ollama)
    elif config.llm.provider == "anthropic":
        api_key = get_anthropic_api_key()
        provider = AnthropicProvider(config.llm.anthropic, api_key)
    else:
        logger.error(f"Unknown provider: {config.llm.provider}")
        sys.exit(1)

    generator = Generator(provider)
    citation_extractor = CitationExtractor()

    # Build filters
    metadata_filter = None
    if args.language or args.type:
        filter_builder = FilterBuilder()
        if args.language:
            filter_builder.language(args.language)
        if args.type:
            filter_builder.type(args.type)
        metadata_filter = filter_builder.build()

    # Run query
    start_time = time.time()

    logger.info(f"\nQuery: {args.query}")
    logger.info("="*60)

    # Retrieval
    retrieval_start = time.time()
    search_results = semantic_search.search(
        query=args.query,
        top_k=args.top_k,
        similarity_threshold=args.threshold,
        filters=metadata_filter,
    )
    retrieval_time = (time.time() - retrieval_start) * 1000

    logger.info(f"\nRetrieved {len(search_results.results)} artifacts in {retrieval_time:.0f}ms")

    if not search_results.results:
        logger.warning("No results found!")
        print("\nNo relevant metadata artifacts found for this query.")
        sys.exit(0)

    # Show retrieved artifacts
    print("\n" + "="*60)
    print("RETRIEVED ARTIFACTS")
    print("="*60)
    for i, result in enumerate(search_results.results, 1):
        artifact = result.artifact
        print(f"\n{i}. {artifact.name} ({artifact.type})")
        print(f"   Similarity: {result.similarity:.3f}")
        print(f"   Language: {artifact.language} | Module: {artifact.module}")
        if artifact.description:
            desc = artifact.description[:100]
            if len(artifact.description) > 100:
                desc += "..."
            print(f"   {desc}")

    # Generation
    print("\n" + "="*60)
    print("GENERATING ANSWER")
    print("="*60)

    generation_start = time.time()
    answer = generator.generate_answer(
        query=args.query,
        search_results=search_results.results,
    )
    generation_time = (time.time() - generation_start) * 1000

    # Extract citations
    citations = citation_extractor.extract_and_validate(answer, search_results.results)
    citation_quality = citation_extractor.get_citation_quality_metrics(answer, search_results.results)

    # Build response
    response = QueryResponse(
        answer=answer,
        query=args.query,
        citations=[CitationInfo.from_citation(c) for c in citations],
        sources=[ArtifactSummary.from_search_result(r) for r in search_results.results],
        search_metadata=SearchMetadata(
            query=args.query,
            top_k=args.top_k,
            similarity_threshold=args.threshold,
            num_results=len(search_results.results),
            filters=metadata_filter.to_dict() if metadata_filter else None,
            retrieval_time_ms=retrieval_time,
        ),
        generation_metadata=GenerationMetadata(
            provider=config.llm.provider,
            model=generator.provider.provider_name(),
            temperature=config.llm.ollama.temperature,
            max_tokens=config.llm.ollama.max_tokens,
            generation_time_ms=generation_time,
        ),
        citation_metrics=CitationMetrics(**citation_quality),
        total_time_ms=(time.time() - start_time) * 1000,
    )

    response.assess_confidence()

    # Output
    if args.markdown:
        print("\n" + response.to_markdown())
    else:
        print("\n" + "="*60)
        print("ANSWER")
        print("="*60)
        print(f"\n{answer}\n")

        print("="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Total time: {response.total_time_ms:.0f}ms")
        print(f"Retrieval: {retrieval_time:.0f}ms | Generation: {generation_time:.0f}ms")
        print(f"Citations: {citation_quality['total_citations']} "
              f"({citation_quality['citation_accuracy']:.1%} accuracy)")
        print(f"Confidence: {response.confidence}")

        if response.warnings:
            print("\nWarnings:")
            for warning in response.warnings:
                print(f"  ⚠️ {warning}")


if __name__ == "__main__":
    main()
