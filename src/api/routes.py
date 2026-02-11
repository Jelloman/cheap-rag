"""FastAPI routes for CHEAP RAG system."""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import load_config, get_anthropic_api_key
from src.embeddings.service import EmbeddingService
from src.generation.citations import CitationExtractor
from src.generation.generator import Generator, OllamaProvider, AnthropicProvider
from src.generation.response import (
    QueryResponse, ErrorResponse, SearchMetadata, GenerationMetadata,
    CitationMetrics, ArtifactSummary, CitationInfo
)
from src.retrieval.filters import FilterBuilder, MetadataFilter
from src.retrieval.semantic_search import SemanticSearch
from src.vectorstore.chroma_store import ChromaVectorStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load configuration
config = load_config()

# Initialize services (singleton pattern)
_embedding_service: Optional[EmbeddingService] = None
_vector_store: Optional[ChromaVectorStore] = None
_semantic_search: Optional[SemanticSearch] = None
_generator: Optional[Generator] = None
_citation_extractor: Optional[CitationExtractor] = None


def get_embedding_service() -> EmbeddingService:
    """Get or create embedding service singleton."""
    global _embedding_service
    if _embedding_service is None:
        logger.info("Initializing EmbeddingService...")
        _embedding_service = EmbeddingService(
            model_name=config.embedding.model_name,
            device=config.embedding.device,
            cache_dir=config.embedding.cache_dir,
            batch_size=config.embedding.batch_size,
        )
    return _embedding_service


def get_vector_store() -> ChromaVectorStore:
    """Get or create vector store singleton."""
    global _vector_store
    if _vector_store is None:
        logger.info("Initializing ChromaVectorStore...")
        _vector_store = ChromaVectorStore(
            persist_directory=config.vectorstore.persist_directory,
            collection_name=config.vectorstore.collection_name,
            distance_metric=config.vectorstore.distance_metric,
        )
    return _vector_store


def get_semantic_search() -> SemanticSearch:
    """Get or create semantic search singleton."""
    global _semantic_search
    if _semantic_search is None:
        logger.info("Initializing SemanticSearch...")
        _semantic_search = SemanticSearch(
            embedding_service=get_embedding_service(),
            vector_store=get_vector_store(),
            default_top_k=config.retrieval.top_k,
            default_similarity_threshold=config.retrieval.similarity_threshold,
        )
    return _semantic_search


def get_generator() -> Generator:
    """Get or create generator singleton."""
    global _generator
    if _generator is None:
        logger.info("Initializing Generator...")
        provider_type = config.llm.provider

        if provider_type == "ollama":
            provider = OllamaProvider(config.llm.ollama)
        elif provider_type == "anthropic":
            api_key = get_anthropic_api_key()
            provider = AnthropicProvider(config.llm.anthropic, api_key)
        else:
            raise ValueError(f"Unknown LLM provider: {provider_type}")

        _generator = Generator(provider)
    return _generator


def get_citation_extractor() -> CitationExtractor:
    """Get or create citation extractor singleton."""
    global _citation_extractor
    if _citation_extractor is None:
        _citation_extractor = CitationExtractor()
    return _citation_extractor


# Pydantic models for requests

class QueryRequest(BaseModel):
    """Request model for query endpoint."""

    query: str = Field(..., description="User's question", min_length=1)
    top_k: Optional[int] = Field(None, description="Number of results to retrieve", ge=1, le=50)
    similarity_threshold: Optional[float] = Field(None, description="Minimum similarity score", ge=0.0, le=1.0)
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")
    temperature: Optional[float] = Field(None, description="LLM temperature", ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, description="Max tokens to generate", ge=100, le=4096)


class IndexStatus(BaseModel):
    """Index status response."""

    collection_name: str
    total_artifacts: int
    embedding_dimension: int
    vector_store_path: str
    last_updated: Optional[datetime] = None


class MetadataBrowseRequest(BaseModel):
    """Request model for browsing metadata."""

    filters: Optional[Dict[str, Any]] = None
    limit: int = Field(20, ge=1, le=100)
    offset: int = Field(0, ge=0)


# Create FastAPI app

app = FastAPI(
    title="CHEAP RAG API",
    description="Semantic search and Q&A over multi-language metadata",
    version="0.1.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.cors_origins if config.api.cors_origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Routes

@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "CHEAP RAG API",
        "version": "0.1.0",
        "status": "running",
        "endpoints": {
            "query": "/api/query",
            "index_status": "/api/index/status",
            "metadata": "/api/metadata",
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check vector store
        vector_store = get_vector_store()
        count = vector_store.count()

        return {
            "status": "healthy",
            "vector_store": "connected",
            "artifacts_count": count,
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """Answer a question using RAG over metadata artifacts.

    Args:
        request: Query request with question and optional filters.

    Returns:
        QueryResponse with answer, sources, and metadata.
    """
    start_time = time.time()

    try:
        logger.info(f"Processing query: '{request.query}'")

        # Parse filters
        metadata_filter = None
        if request.filters:
            # Convert dict to MetadataFilter
            metadata_filter = MetadataFilter(**request.filters)

        # Retrieval
        retrieval_start = time.time()
        semantic_search = get_semantic_search()
        search_results = semantic_search.search(
            query=request.query,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold,
            filters=metadata_filter,
        )
        retrieval_time = (time.time() - retrieval_start) * 1000

        logger.info(f"Retrieved {len(search_results.results)} artifacts in {retrieval_time:.0f}ms")

        if not search_results.results:
            # No results found
            return QueryResponse(
                answer="I don't know based on the provided context. No relevant metadata artifacts were found for this query.",
                query=request.query,
                search_metadata=SearchMetadata(
                    query=request.query,
                    top_k=request.top_k or config.retrieval.top_k,
                    similarity_threshold=request.similarity_threshold or config.retrieval.similarity_threshold,
                    num_results=0,
                    filters=request.filters,
                    retrieval_time_ms=retrieval_time,
                ),
                total_time_ms=(time.time() - start_time) * 1000,
                warnings=["No artifacts matched the query and filters"],
            )

        # Generation
        generation_start = time.time()
        generator = get_generator()
        answer = generator.generate_answer(
            query=request.query,
            search_results=search_results.results,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        generation_time = (time.time() - generation_start) * 1000

        logger.info(f"Generated answer in {generation_time:.0f}ms")

        # Extract and validate citations
        citation_extractor = get_citation_extractor()
        citations = citation_extractor.extract_and_validate(answer, search_results.results)
        citation_quality = citation_extractor.get_citation_quality_metrics(answer, search_results.results)

        # Build response
        response = QueryResponse(
            answer=answer,
            query=request.query,
            citations=[CitationInfo.from_citation(c) for c in citations],
            sources=[ArtifactSummary.from_search_result(r) for r in search_results.results],
            search_metadata=SearchMetadata(
                query=request.query,
                top_k=search_results.top_k,
                similarity_threshold=search_results.similarity_threshold,
                num_results=len(search_results.results),
                filters=request.filters,
                retrieval_time_ms=retrieval_time,
            ),
            generation_metadata=GenerationMetadata(
                provider=config.llm.provider,
                model=generator.provider.provider_name(),
                temperature=request.temperature or config.llm.ollama.temperature,
                max_tokens=request.max_tokens or config.llm.ollama.max_tokens,
                generation_time_ms=generation_time,
            ),
            citation_metrics=CitationMetrics(**citation_quality),
            total_time_ms=(time.time() - start_time) * 1000,
        )

        # Assess confidence
        response.assess_confidence()

        # Add warnings
        if citation_quality["has_hallucinations"]:
            response.add_warning("Answer contains citations to artifacts not in search results")
        if citation_quality["citation_coverage"] < 0.5:
            response.add_warning("Low citation coverage - not all retrieved artifacts were used")

        logger.info(f"Query completed in {response.total_time_ms:.0f}ms")

        return response

    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@app.get("/api/index/status", response_model=IndexStatus)
async def index_status() -> IndexStatus:
    """Get indexing statistics and status.

    Returns:
        IndexStatus with collection info.
    """
    try:
        vector_store = get_vector_store()
        embedding_service = get_embedding_service()

        return IndexStatus(
            collection_name=config.vectorstore.collection_name,
            total_artifacts=vector_store.count(),
            embedding_dimension=embedding_service.get_dimension(),
            vector_store_path=config.vectorstore.persist_directory,
        )

    except Exception as e:
        logger.error(f"Index status check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/index/rebuild")
async def rebuild_index():
    """Trigger full reindexing (placeholder).

    Note: Actual reindexing should be done via CLI scripts for Phase 1.
    """
    return {
        "message": "Index rebuild not implemented in Phase 1",
        "recommendation": "Use scripts/index_metadata.py to rebuild the index",
    }


@app.get("/api/metadata")
async def browse_metadata(
    language: Optional[str] = Query(None, description="Filter by language"),
    type: Optional[str] = Query(None, description="Filter by artifact type"),
    source_type: Optional[str] = Query(None, description="Filter by source type"),
    limit: int = Query(20, ge=1, le=100, description="Number of results"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
):
    """Browse metadata artifacts with filters and pagination.

    Args:
        language: Filter by language (e.g., "java", "postgresql").
        type: Filter by artifact type (e.g., "table", "class").
        source_type: Filter by source (e.g., "database", "code").
        limit: Number of results to return.
        offset: Offset for pagination.

    Returns:
        List of artifact summaries.
    """
    try:
        # For Phase 1, return a simple implementation
        # Full implementation would query ChromaDB with filters and pagination
        vector_store = get_vector_store()

        # Build filters
        filter_builder = FilterBuilder()
        if language:
            filter_builder.language(language)
        if type:
            filter_builder.type(type)
        if source_type:
            filter_builder.source_type(source_type)

        metadata_filter = filter_builder.build()

        # Note: ChromaDB doesn't have native pagination
        # For Phase 1, this is a placeholder
        return {
            "message": "Metadata browsing with pagination not fully implemented in Phase 1",
            "total_artifacts": vector_store.count(),
            "filters": metadata_filter.to_dict() if metadata_filter else {},
            "limit": limit,
            "offset": offset,
        }

    except Exception as e:
        logger.error(f"Metadata browse failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Startup/shutdown events

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting CHEAP RAG API...")
    logger.info(f"Configuration profile: {config.llm.provider}")

    # Initialize services
    get_embedding_service()
    get_vector_store()
    get_semantic_search()
    get_generator()

    logger.info("All services initialized successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down CHEAP RAG API...")


# Error handlers

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return ErrorResponse(
        error=exc.detail,
        error_type="HTTPException",
        query=None,
    ).to_dict()


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return ErrorResponse(
        error=str(exc),
        error_type=type(exc).__name__,
        query=None,
    ).to_dict()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=config.api.host,
        port=config.api.port,
        log_level="info",
    )
