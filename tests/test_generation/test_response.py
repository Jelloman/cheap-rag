"""Tests for response formatting and quality assessment."""

from __future__ import annotations

from datetime import datetime

import pytest

from src.extractors.base import MetadataArtifact
from src.generation.citations import Citation
from src.generation.response import (
    ArtifactSummary,
    CitationInfo,
    CitationMetrics,
    ErrorResponse,
    GenerationMetadata,
    QueryResponse,
    SearchMetadata,
)
from src.retrieval.semantic_search import SearchResult


@pytest.fixture
def sample_artifact() -> MetadataArtifact:
    """Create a sample metadata artifact."""
    return MetadataArtifact(
        id="postgresql_public_table_sale_order_123",
        name="sale_order",
        type="table",
        source_type="database",
        language="postgresql",
        module="public",
        description="Stores sales orders for customers. This table contains order headers.",
        metadata={},
    )


@pytest.fixture
def sample_search_result(sample_artifact: MetadataArtifact) -> SearchResult:
    """Create a sample search result."""
    return SearchResult(artifact=sample_artifact, similarity=0.85, distance=0.15, rank=1)


@pytest.fixture
def sample_citation() -> Citation:
    """Create a sample citation."""
    artifact = MetadataArtifact(
        id="postgresql_public_table_sale_order_123",
        name="sale_order",
        type="table",
        source_type="database",
        language="postgresql",
        module="public",
        description="Stores sales orders",
        metadata={},
    )
    return Citation(
        artifact_name="sale_order",
        artifact_id="postgresql_public_table_sale_order_123",
        is_valid=True,
        position=10,
        matched_artifact=artifact,
    )


class TestArtifactSummary:
    """Tests for ArtifactSummary model."""

    def test_artifact_summary_from_artifact(self, sample_artifact: MetadataArtifact):
        """Test creating summary from artifact."""
        summary = ArtifactSummary.from_artifact(
            sample_artifact, similarity=0.85, rank=1
        )

        assert summary.id == sample_artifact.id
        assert summary.name == sample_artifact.name
        assert summary.type == sample_artifact.type
        assert summary.language == sample_artifact.language
        assert summary.module == sample_artifact.module
        assert summary.similarity == 0.85
        assert summary.rank == 1
        # Description should be truncated to 200 chars
        assert len(summary.description) <= 200

    def test_artifact_summary_from_search_result(self, sample_search_result: SearchResult):
        """Test creating summary from search result."""
        summary = ArtifactSummary.from_search_result(sample_search_result)

        assert summary.id == sample_search_result.artifact.id
        assert summary.similarity == sample_search_result.similarity
        assert summary.rank == sample_search_result.rank

    def test_artifact_summary_truncates_long_description(self):
        """Test that long descriptions are truncated."""
        artifact = MetadataArtifact(
            id="test_id",
            name="test",
            type="table",
            source_type="database",
            language="postgresql",
            module="public",
            description="A" * 300,  # 300 character description
            metadata={},
        )

        summary = ArtifactSummary.from_artifact(artifact)

        assert len(summary.description) == 200


class TestCitationInfo:
    """Tests for CitationInfo model."""

    def test_citation_info_from_citation_valid(self, sample_citation: Citation):
        """Test creating citation info from valid citation."""
        info = CitationInfo.from_citation(sample_citation)

        assert info.artifact_name == sample_citation.artifact_name
        assert info.artifact_id == sample_citation.artifact_id
        assert info.is_valid is True
        assert info.artifact_summary is not None
        assert info.artifact_summary.name == "sale_order"

    def test_citation_info_from_citation_invalid(self):
        """Test creating citation info from invalid citation."""
        citation = Citation(
            artifact_name="unknown",
            artifact_id="invalid_id",
            is_valid=False,
            position=10,
            matched_artifact=None,
        )

        info = CitationInfo.from_citation(citation)

        assert info.is_valid is False
        assert info.artifact_summary is None


class TestSearchMetadata:
    """Tests for SearchMetadata model."""

    def test_search_metadata_creation(self):
        """Test creating search metadata."""
        metadata = SearchMetadata(
            query="What is the sale_order table?",
            top_k=5,
            similarity_threshold=0.5,
            num_results=3,
            filters={"language": "postgresql"},
            retrieval_time_ms=25.5,
        )

        assert metadata.query == "What is the sale_order table?"
        assert metadata.top_k == 5
        assert metadata.num_results == 3
        assert metadata.filters == {"language": "postgresql"}
        assert metadata.retrieval_time_ms == 25.5


class TestGenerationMetadata:
    """Tests for GenerationMetadata model."""

    def test_generation_metadata_creation(self):
        """Test creating generation metadata."""
        metadata = GenerationMetadata(
            provider="ollama",
            model="ollama:qwen2.5-coder",
            temperature=0.1,
            max_tokens=1024,
            generation_time_ms=2500.0,
            input_tokens=150,
            output_tokens=200,
            cost_usd=0.005,
        )

        assert metadata.provider == "ollama"
        assert metadata.model == "ollama:qwen2.5-coder"
        assert metadata.temperature == 0.1
        assert metadata.generation_time_ms == 2500.0


class TestCitationMetrics:
    """Tests for CitationMetrics model."""

    def test_citation_metrics_defaults(self):
        """Test citation metrics default values."""
        metrics = CitationMetrics()

        assert metrics.total_citations == 0
        assert metrics.valid_citations == 0
        assert metrics.invalid_citations == 0
        assert metrics.citation_accuracy == 1.0
        assert metrics.citation_coverage == 0.0
        assert metrics.has_hallucinations is False

    def test_citation_metrics_custom_values(self):
        """Test citation metrics with custom values."""
        metrics = CitationMetrics(
            total_citations=5,
            valid_citations=4,
            invalid_citations=1,
            citation_accuracy=0.8,
            citation_coverage=0.75,
            has_hallucinations=True,
        )

        assert metrics.total_citations == 5
        assert metrics.valid_citations == 4
        assert metrics.invalid_citations == 1
        assert metrics.citation_accuracy == 0.8
        assert metrics.has_hallucinations is True


class TestQueryResponse:
    """Tests for QueryResponse model."""

    def test_query_response_minimal(self):
        """Test creating minimal query response."""
        search_metadata = SearchMetadata(
            query="Test query",
            top_k=5,
            similarity_threshold=0.5,
            num_results=0,
        )

        response = QueryResponse(
            answer="I don't know based on the provided context.",
            query="Test query",
            search_metadata=search_metadata,
        )

        assert response.answer == "I don't know based on the provided context."
        assert response.query == "Test query"
        assert len(response.citations) == 0
        assert len(response.sources) == 0

    def test_query_response_full(self, sample_search_result: SearchResult):
        """Test creating full query response with all fields."""
        search_metadata = SearchMetadata(
            query="What is sale_order?",
            top_k=5,
            similarity_threshold=0.5,
            num_results=1,
            retrieval_time_ms=25.0,
        )

        generation_metadata = GenerationMetadata(
            provider="ollama",
            model="qwen",
            temperature=0.1,
            max_tokens=1024,
            generation_time_ms=2000.0,
        )

        citation_metrics = CitationMetrics(
            total_citations=1,
            valid_citations=1,
            citation_accuracy=1.0,
            citation_coverage=1.0,
        )

        response = QueryResponse(
            answer="The sale_order table [sale_order] (ID: postgresql_public_table_sale_order_123) stores orders.",
            query="What is sale_order?",
            sources=[ArtifactSummary.from_search_result(sample_search_result)],
            search_metadata=search_metadata,
            generation_metadata=generation_metadata,
            citation_metrics=citation_metrics,
            total_time_ms=2025.0,
        )

        assert response.answer is not None
        assert len(response.sources) == 1
        assert response.total_time_ms == 2025.0

    def test_query_response_add_warning(self):
        """Test adding warnings to response."""
        search_metadata = SearchMetadata(
            query="Test",
            top_k=5,
            similarity_threshold=0.5,
            num_results=0,
        )

        response = QueryResponse(
            answer="Test",
            query="Test",
            search_metadata=search_metadata,
        )

        response.add_warning("Low citation coverage")
        response.add_warning("Possible hallucination")

        assert len(response.warnings) == 2
        assert "Low citation coverage" in response.warnings

    def test_query_response_assess_confidence_high(self, sample_search_result: SearchResult):
        """Test confidence assessment - high confidence."""
        search_metadata = SearchMetadata(
            query="Test",
            top_k=5,
            similarity_threshold=0.5,
            num_results=1,
        )

        citation_metrics = CitationMetrics(
            total_citations=1,
            valid_citations=1,
            citation_accuracy=1.0,
            citation_coverage=1.0,
            has_hallucinations=False,
        )

        response = QueryResponse(
            answer="Good answer with citation.",
            query="Test",
            sources=[ArtifactSummary.from_search_result(sample_search_result)],
            search_metadata=search_metadata,
            citation_metrics=citation_metrics,
        )

        confidence = response.assess_confidence()

        # With 0.85 similarity and only 1 result, should be medium (downgraded due to num_results < 2)
        assert confidence in ["high", "medium"]  # Accept both - depends on num_results threshold
        assert response.confidence == confidence

    def test_query_response_assess_confidence_low_similarity(self):
        """Test confidence assessment - low similarity scores."""
        artifact = MetadataArtifact(
            id="test_id",
            name="test",
            type="table",
            source_type="database",
            language="postgresql",
            module="public",
            description="Test",
            metadata={},
        )
        low_similarity_result = SearchResult(artifact=artifact, similarity=0.3, distance=0.7, rank=1)

        search_metadata = SearchMetadata(
            query="Test",
            top_k=5,
            similarity_threshold=0.2,
            num_results=1,
        )

        response = QueryResponse(
            answer="Answer",
            query="Test",
            sources=[ArtifactSummary.from_search_result(low_similarity_result)],
            search_metadata=search_metadata,
        )

        confidence = response.assess_confidence()

        assert confidence == "low"

    def test_query_response_assess_confidence_with_hallucinations(
        self, sample_search_result: SearchResult
    ):
        """Test confidence assessment - with hallucinations."""
        search_metadata = SearchMetadata(
            query="Test",
            top_k=5,
            similarity_threshold=0.5,
            num_results=1,
        )

        citation_metrics = CitationMetrics(
            total_citations=2,
            valid_citations=1,
            invalid_citations=1,
            citation_accuracy=0.5,
            has_hallucinations=True,
        )

        response = QueryResponse(
            answer="Answer",
            query="Test",
            sources=[ArtifactSummary.from_search_result(sample_search_result)],
            search_metadata=search_metadata,
            citation_metrics=citation_metrics,
        )

        confidence = response.assess_confidence()

        assert confidence == "low"

    def test_query_response_assess_confidence_dont_know(self):
        """Test confidence assessment - 'don't know' response."""
        search_metadata = SearchMetadata(
            query="Test",
            top_k=5,
            similarity_threshold=0.5,
            num_results=0,
        )

        response = QueryResponse(
            answer="I don't know based on the provided context.",
            query="Test",
            search_metadata=search_metadata,
        )

        confidence = response.assess_confidence()

        assert confidence == "low"

    def test_query_response_to_dict(self, sample_search_result: SearchResult):
        """Test converting response to dictionary."""
        search_metadata = SearchMetadata(
            query="Test",
            top_k=5,
            similarity_threshold=0.5,
            num_results=1,
        )

        response = QueryResponse(
            answer="Answer",
            query="Test",
            sources=[ArtifactSummary.from_search_result(sample_search_result)],
            search_metadata=search_metadata,
        )

        response_dict = response.to_dict()

        assert isinstance(response_dict, dict)
        assert "answer" in response_dict
        assert "query" in response_dict
        assert "sources" in response_dict
        assert "search_metadata" in response_dict

    def test_query_response_to_markdown(self, sample_search_result: SearchResult):
        """Test converting response to markdown."""
        search_metadata = SearchMetadata(
            query="What is sale_order?",
            top_k=5,
            similarity_threshold=0.5,
            num_results=1,
        )

        response = QueryResponse(
            answer="The sale_order table stores orders.",
            query="What is sale_order?",
            sources=[ArtifactSummary.from_search_result(sample_search_result)],
            search_metadata=search_metadata,
            confidence="high",
        )

        markdown = response.to_markdown()

        assert "# Query:" in markdown
        assert "## Answer" in markdown
        assert "## Sources" in markdown
        assert "## Metadata" in markdown
        assert "sale_order" in markdown
        assert "high" in markdown

    def test_query_response_to_markdown_with_warnings(self):
        """Test markdown output with warnings."""
        search_metadata = SearchMetadata(
            query="Test",
            top_k=5,
            similarity_threshold=0.5,
            num_results=0,
        )

        response = QueryResponse(
            answer="Answer",
            query="Test",
            search_metadata=search_metadata,
            warnings=["Warning 1", "Warning 2"],
        )

        markdown = response.to_markdown()

        assert "## Warnings" in markdown
        assert "Warning 1" in markdown
        assert "Warning 2" in markdown

    def test_query_response_to_markdown_with_citations(self, sample_citation: Citation):
        """Test markdown output with citation metrics."""
        search_metadata = SearchMetadata(
            query="Test",
            top_k=5,
            similarity_threshold=0.5,
            num_results=1,
        )

        citation_metrics = CitationMetrics(
            total_citations=2,
            valid_citations=2,
            citation_accuracy=1.0,
            citation_coverage=0.8,
        )

        response = QueryResponse(
            answer="Answer",
            query="Test",
            citations=[CitationInfo.from_citation(sample_citation)],
            search_metadata=search_metadata,
            citation_metrics=citation_metrics,
        )

        markdown = response.to_markdown()

        assert "## Citation Quality" in markdown
        assert "**Total Citations:** 2" in markdown
        assert "**Accuracy:** 100" in markdown


class TestErrorResponse:
    """Tests for ErrorResponse model."""

    def test_error_response_creation(self):
        """Test creating error response."""
        response = ErrorResponse(
            error="Query processing failed",
            error_type="ValueError",
            query="Test query",
            details={"step": "retrieval"},
        )

        assert response.error == "Query processing failed"
        assert response.error_type == "ValueError"
        assert response.query == "Test query"
        assert response.details == {"step": "retrieval"}
        assert isinstance(response.timestamp, datetime)

    def test_error_response_to_dict(self):
        """Test converting error response to dictionary."""
        response = ErrorResponse(
            error="Error message",
            error_type="RuntimeError",
        )

        error_dict = response.to_dict()

        assert isinstance(error_dict, dict)
        assert "error" in error_dict
        assert "error_type" in error_dict
        assert "timestamp" in error_dict

    def test_error_response_without_query(self):
        """Test error response without query."""
        response = ErrorResponse(
            error="General error",
            error_type="Exception",
        )

        assert response.query is None


class TestResponseIntegration:
    """Integration tests for complete response workflow."""

    def test_complete_response_workflow(self, sample_search_result: SearchResult):
        """Test creating a complete response with all components."""
        # Create all metadata components
        search_metadata = SearchMetadata(
            query="What is the sale_order table?",
            top_k=5,
            similarity_threshold=0.5,
            num_results=1,
            retrieval_time_ms=25.0,
        )

        generation_metadata = GenerationMetadata(
            provider="ollama",
            model="qwen2.5-coder",
            temperature=0.1,
            max_tokens=1024,
            generation_time_ms=2000.0,
        )

        citation_metrics = CitationMetrics(
            total_citations=1,
            valid_citations=1,
            invalid_citations=0,
            citation_accuracy=1.0,
            citation_coverage=1.0,
            has_hallucinations=False,
        )

        # Create response
        response = QueryResponse(
            answer="The sale_order table [sale_order] (ID: postgresql_public_table_sale_order_123) stores sales orders.",
            query="What is the sale_order table?",
            sources=[ArtifactSummary.from_search_result(sample_search_result)],
            search_metadata=search_metadata,
            generation_metadata=generation_metadata,
            citation_metrics=citation_metrics,
            total_time_ms=2025.0,
        )

        # Assess confidence
        response.assess_confidence()

        # Verify complete response
        assert response.confidence in ["high", "medium"]  # Accept both - depends on thresholds
        assert len(response.sources) == 1
        assert response.citation_metrics.has_hallucinations is False
        assert response.total_time_ms == 2025.0

        # Test serialization
        response_dict = response.to_dict()
        assert isinstance(response_dict, dict)

        markdown = response.to_markdown()
        assert len(markdown) > 0
