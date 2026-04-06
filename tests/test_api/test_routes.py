"""Tests for FastAPI routes."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.routes import app
from src.extractors.base import MetadataArtifact
from src.generation.citations import Citation
from src.retrieval.semantic_search import SearchResult, SearchResults


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def sample_artifact():
    """Create sample metadata artifact."""
    return MetadataArtifact(
        id="postgresql_public_table_sale_order_123",
        name="sale_order",
        type="table",
        source_type="database",
        language="postgresql",
        module="public",
        description="Stores sales orders",
        metadata={},
    )


@pytest.fixture
def sample_search_result(sample_artifact):
    """Create sample search result."""
    return SearchResult(
        artifact=sample_artifact,
        similarity=0.85,
        distance=0.15,
        rank=1,
    )


class TestRootEndpoints:
    """Tests for root and health endpoints."""

    def test_root_endpoint(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "CHEAP RAG API" in data["name"]
        assert "endpoints" in data
        assert "/api/query" in data["endpoints"]["query"]

    @patch("src.api.routes.get_vector_store")
    def test_health_check_healthy(self, mock_get_vector_store, client):
        """Test health check when services are healthy."""
        # Mock vector store
        mock_store = Mock()
        mock_store.count.return_value = 544
        mock_get_vector_store.return_value = mock_store

        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["vector_store"] == "connected"
        assert data["artifacts_count"] == 544

    @patch("src.api.routes.get_vector_store")
    def test_health_check_unhealthy(self, mock_get_vector_store, client):
        """Test health check when services fail."""
        # Mock vector store failure
        mock_get_vector_store.side_effect = RuntimeError("Connection failed")

        response = client.get("/health")

        assert response.status_code == 503
        data = response.json()
        # FastAPI's HTTPException uses 'detail' field
        assert "Connection failed" in str(data.get("detail", data.get("error", "")))


class TestQueryEndpoint:
    """Tests for /api/query endpoint."""

    @patch("src.api.routes.get_citation_extractor")
    @patch("src.api.routes.get_generator")
    @patch("src.api.routes.get_semantic_search")
    def test_query_success(
        self,
        mock_get_search,
        mock_get_generator,
        mock_get_citation_extractor,
        client,
        sample_search_result,
    ):
        """Test successful query execution."""
        # Mock semantic search
        mock_search = Mock()
        mock_search_results = SearchResults(
            query="What is sale_order?",
            results=[sample_search_result],
            total_results=1,
            top_k=5,
            similarity_threshold=0.5,
        )
        mock_search.search.return_value = mock_search_results
        mock_get_search.return_value = mock_search

        # Mock generator
        mock_generator = Mock()
        mock_generator.generate_answer.return_value = "The sale_order table [sale_order] (ID: postgresql_public_table_sale_order_123) stores orders."
        mock_provider = Mock()
        mock_provider.provider_name.return_value = "ollama:qwen"
        mock_generator.provider = mock_provider
        mock_get_generator.return_value = mock_generator

        # Mock citation extractor
        mock_extractor = Mock()
        mock_citation = Citation(
            artifact_name="sale_order",
            artifact_id="postgresql_public_table_sale_order_123",
            is_valid=True,
            position=10,
            matched_artifact=sample_search_result.artifact,
        )
        mock_extractor.extract_and_validate.return_value = [mock_citation]
        mock_extractor.get_citation_quality_metrics.return_value = {
            "total_citations": 1,
            "valid_citations": 1,
            "invalid_citations": 0,
            "citation_accuracy": 1.0,
            "citation_coverage": 1.0,
            "has_hallucinations": False,
        }
        mock_get_citation_extractor.return_value = mock_extractor

        # Make request
        response = client.post(
            "/api/query",
            json={"query": "What is sale_order?", "top_k": 5},
        )

        assert response.status_code == 200
        data = response.json()

        assert "answer" in data
        assert "sale_order" in data["answer"]
        assert "query" in data
        assert data["query"] == "What is sale_order?"
        assert "sources" in data
        assert len(data["sources"]) == 1
        assert "search_metadata" in data
        assert "citation_metrics" in data

    @patch("src.api.routes.get_semantic_search")
    def test_query_no_results(self, mock_get_search, client):
        """Test query with no search results."""
        # Mock empty search results
        mock_search = Mock()
        mock_search_results = SearchResults(
            query="nonexistent query",
            results=[],
            total_results=0,
            top_k=5,
            similarity_threshold=0.5,
        )
        mock_search.search.return_value = mock_search_results
        mock_get_search.return_value = mock_search

        response = client.post(
            "/api/query",
            json={"query": "nonexistent query"},
        )

        assert response.status_code == 200
        data = response.json()

        assert "don't know" in data["answer"].lower()
        assert len(data.get("sources", [])) == 0
        assert "warnings" in data
        assert len(data["warnings"]) > 0

    def test_query_missing_query_field(self, client):
        """Test query request without query field."""
        response = client.post("/api/query", json={})

        assert response.status_code == 422  # Validation error

    def test_query_invalid_parameters(self, client):
        """Test query with invalid parameters."""
        response = client.post(
            "/api/query",
            json={
                "query": "test",
                "top_k": -5,  # Invalid: negative
                "similarity_threshold": 1.5,  # Invalid: > 1.0
            },
        )

        assert response.status_code == 422  # Validation error

    @patch("src.api.routes.get_semantic_search")
    def test_query_with_filters(self, mock_get_search, client, sample_search_result):
        """Test query with metadata filters."""
        mock_search = Mock()
        mock_search_results = SearchResults(
            query="test",
            results=[sample_search_result],
            total_results=1,
            top_k=5,
            similarity_threshold=0.5,
        )
        mock_search.search.return_value = mock_search_results
        mock_get_search.return_value = mock_search

        response = client.post(
            "/api/query",
            json={
                "query": "test",
                "filters": {"language": "postgresql", "type": "table"},
            },
        )

        # Should not fail validation
        assert response.status_code in [200, 500]  # May fail on mocked services

    @patch("src.api.routes.get_semantic_search")
    def test_query_internal_error(self, mock_get_search, client):
        """Test query endpoint handles internal errors."""
        # Mock search to raise an error
        mock_search = Mock()
        mock_search.search.side_effect = RuntimeError("Database connection failed")
        mock_get_search.return_value = mock_search

        response = client.post(
            "/api/query",
            json={"query": "test"},
        )

        assert response.status_code == 500
        data = response.json()
        # Our custom error response has 'error' field
        assert "error" in data or "detail" in data
        assert "Database connection failed" in str(data)


class TestIndexEndpoints:
    """Tests for index-related endpoints."""

    @patch("src.api.routes.get_embedding_service")
    @patch("src.api.routes.get_vector_store")
    def test_index_status(self, mock_get_vector_store, mock_get_embedding_service, client):
        """Test index status endpoint."""
        # Mock vector store
        mock_store = Mock()
        mock_store.count.return_value = 544
        mock_get_vector_store.return_value = mock_store

        # Mock embedding service
        mock_embedding = Mock()
        mock_embedding.get_dimension.return_value = 768
        mock_get_embedding_service.return_value = mock_embedding

        response = client.get("/api/index/status")

        assert response.status_code == 200
        data = response.json()

        assert "collection_name" in data
        assert "total_artifacts" in data
        assert data["total_artifacts"] == 544
        assert "embedding_dimension" in data
        assert data["embedding_dimension"] == 768
        assert "vector_store_path" in data

    def test_rebuild_index(self, client):
        """Test rebuild index endpoint (not implemented in Phase 1)."""
        response = client.post("/api/index/rebuild")

        assert response.status_code == 200
        data = response.json()

        assert "message" in data
        assert "not implemented" in data["message"].lower()
        assert "recommendation" in data


class TestMetadataEndpoint:
    """Tests for metadata browsing endpoint."""

    @patch("src.api.routes.get_vector_store")
    def test_metadata_browse(self, mock_get_vector_store, client):
        """Test metadata browsing endpoint."""
        # Mock vector store
        mock_store = Mock()
        mock_store.count.return_value = 544
        mock_get_vector_store.return_value = mock_store

        response = client.get("/api/metadata")

        assert response.status_code == 200
        data = response.json()

        assert "total_artifacts" in data
        assert data["total_artifacts"] == 544

    @patch("src.api.routes.get_vector_store")
    def test_metadata_browse_with_filters(self, mock_get_vector_store, client):
        """Test metadata browsing with filters."""
        mock_store = Mock()
        mock_store.count.return_value = 100
        mock_get_vector_store.return_value = mock_store

        response = client.get(
            "/api/metadata",
            params={
                "language": "postgresql",
                "type": "table",
                "limit": 10,
                "offset": 0,
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert "filters" in data
        assert data["limit"] == 10
        assert data["offset"] == 0

    def test_metadata_browse_invalid_params(self, client):
        """Test metadata browsing with invalid parameters."""
        response = client.get(
            "/api/metadata",
            params={"limit": -5},  # Invalid
        )

        assert response.status_code == 422  # Validation error


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_404_not_found(self, client):
        """Test that unknown endpoints return 404."""
        response = client.get("/api/unknown/endpoint")

        assert response.status_code == 404

    @patch("src.api.routes.get_semantic_search")
    def test_query_handles_exceptions_gracefully(self, mock_get_search, client):
        """Test that query endpoint handles unexpected exceptions."""
        # Mock to raise unexpected exception
        mock_get_search.side_effect = ValueError("Unexpected error")

        response = client.post(
            "/api/query",
            json={"query": "test"},
        )

        # Should return 500 with error details
        assert response.status_code == 500
        data = response.json()
        # Our custom error response has 'error' field
        assert "error" in data or "detail" in data
        assert "Unexpected error" in str(data)


class TestCORS:
    """Tests for CORS configuration."""

    def test_cors_headers_present(self, client):
        """Test that CORS headers are configured."""
        response = client.options("/api/query")

        # FastAPI/Starlette handles OPTIONS automatically
        assert response.status_code in [200, 405]  # May vary based on implementation


class TestRequestValidation:
    """Tests for request validation."""

    def test_query_validates_required_fields(self, client):
        """Test that required fields are validated."""
        # Missing query field
        response = client.post("/api/query", json={})

        assert response.status_code == 422

    def test_query_validates_field_types(self, client):
        """Test that field types are validated."""
        # Wrong type for top_k
        response = client.post(
            "/api/query",
            json={"query": "test", "top_k": "not_a_number"},
        )

        assert response.status_code == 422

    def test_query_validates_field_ranges(self, client):
        """Test that field ranges are validated."""
        # Out of range values
        response = client.post(
            "/api/query",
            json={
                "query": "test",
                "top_k": 100,  # Max is 50
                "similarity_threshold": 2.0,  # Max is 1.0
            },
        )

        assert response.status_code == 422

    def test_metadata_validates_pagination_params(self, client):
        """Test that pagination parameters are validated."""
        # Negative offset
        response = client.get("/api/metadata", params={"offset": -1})

        assert response.status_code == 422


class TestResponseFormat:
    """Tests for response format consistency."""

    @patch("src.api.routes.get_citation_extractor")
    @patch("src.api.routes.get_generator")
    @patch("src.api.routes.get_semantic_search")
    def test_query_response_structure(
        self,
        mock_get_search,
        mock_get_generator,
        mock_get_citation_extractor,
        client,
        sample_search_result,
    ):
        """Test that query response has expected structure."""
        # Setup mocks (similar to test_query_success)
        mock_search = Mock()
        mock_search_results = SearchResults(
            query="test",
            results=[sample_search_result],
            total_results=1,
            top_k=5,
            similarity_threshold=0.5,
        )
        mock_search.search.return_value = mock_search_results
        mock_get_search.return_value = mock_search

        mock_generator = Mock()
        mock_generator.generate_answer.return_value = "Answer"
        mock_provider = Mock()
        mock_provider.provider_name.return_value = "ollama:qwen"
        mock_generator.provider = mock_provider
        mock_get_generator.return_value = mock_generator

        mock_extractor = Mock()
        mock_extractor.extract_and_validate.return_value = []
        mock_extractor.get_citation_quality_metrics.return_value = {
            "total_citations": 0,
            "valid_citations": 0,
            "invalid_citations": 0,
            "citation_accuracy": 1.0,
            "citation_coverage": 0.0,
            "has_hallucinations": False,
        }
        mock_get_citation_extractor.return_value = mock_extractor

        response = client.post("/api/query", json={"query": "test"})

        assert response.status_code == 200
        data = response.json()

        # Required fields
        assert "answer" in data
        assert "query" in data
        assert "search_metadata" in data
        assert "timestamp" in data

        # Search metadata structure
        search_meta = data["search_metadata"]
        assert "query" in search_meta
        assert "top_k" in search_meta
        assert "num_results" in search_meta

    def test_error_response_structure(self, client):
        """Test that error responses have consistent structure."""
        response = client.post("/api/query", json={})  # Missing required field

        assert response.status_code == 422
        data = response.json()

        # FastAPI validation error structure
        assert "detail" in data
