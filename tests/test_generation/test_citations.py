"""Tests for citation extraction and validation."""

from __future__ import annotations

import pytest

from src.extractors.base import MetadataArtifact
from src.generation.citations import Citation, CitationExtractor, format_sources_list
from src.retrieval.semantic_search import SearchResult


@pytest.fixture
def citation_extractor() -> CitationExtractor:
    """Create a citation extractor instance."""
    return CitationExtractor()


@pytest.fixture
def sample_artifacts() -> list[MetadataArtifact]:
    """Create sample metadata artifacts."""
    return [
        MetadataArtifact(
            id="postgresql_public_table_sale_order_123",
            name="sale_order",
            type="table",
            source_type="database",
            language="postgresql",
            module="public",
            description="Stores sales orders for customers",
            metadata={},
        ),
        MetadataArtifact(
            id="postgresql_public_column_sale_order_id_456",
            name="id",
            type="column",
            source_type="database",
            language="postgresql",
            module="public",
            description="Primary key",
            metadata={"table_name": "sale_order"},
        ),
        MetadataArtifact(
            id="postgresql_public_column_sale_order_partner_id_789",
            name="partner_id",
            type="column",
            source_type="database",
            language="postgresql",
            module="public",
            description="Foreign key to partner",
            metadata={"table_name": "sale_order"},
        ),
    ]


@pytest.fixture
def sample_search_results(sample_artifacts: list[MetadataArtifact]) -> list[SearchResult]:
    """Create sample search results."""
    return [
        SearchResult(
            artifact=artifact, similarity=0.9 - i * 0.1, distance=0.1 + i * 0.1, rank=i + 1
        )
        for i, artifact in enumerate(sample_artifacts)
    ]


class TestCitationExtraction:
    """Tests for citation extraction from text."""

    def test_extract_single_citation(self, citation_extractor: CitationExtractor):
        """Test extracting a single citation."""
        answer = "The sale_order table [sale_order] (ID: postgresql_public_table_sale_order_123) stores orders."

        citations = citation_extractor.extract_citations(answer)

        assert len(citations) == 1
        assert citations[0].artifact_name == "sale_order"
        assert citations[0].artifact_id == "postgresql_public_table_sale_order_123"
        assert citations[0].position == answer.find("[sale_order]")

    def test_extract_multiple_citations(self, citation_extractor: CitationExtractor):
        """Test extracting multiple citations."""
        answer = (
            "The sale_order table [sale_order] (ID: postgresql_public_table_sale_order_123) "
            "has an id column [id] (ID: postgresql_public_column_sale_order_id_456) "
            "and a partner_id column [partner_id] (ID: postgresql_public_column_sale_order_partner_id_789)."
        )

        citations = citation_extractor.extract_citations(answer)

        assert len(citations) == 3
        assert citations[0].artifact_name == "sale_order"
        assert citations[1].artifact_name == "id"
        assert citations[2].artifact_name == "partner_id"

    def test_extract_no_citations(self, citation_extractor: CitationExtractor):
        """Test extracting from text with no citations."""
        answer = "This is a response without any citations."

        citations = citation_extractor.extract_citations(answer)

        assert len(citations) == 0

    def test_extract_citation_with_whitespace(self, citation_extractor: CitationExtractor):
        """Test extracting citations with varying whitespace."""
        answer = "Test [artifact]  (ID:  some_id_123  ) more text."

        citations = citation_extractor.extract_citations(answer)

        assert len(citations) == 1
        assert citations[0].artifact_name == "artifact"
        assert citations[0].artifact_id == "some_id_123"

    def test_extract_citation_with_special_chars(self, citation_extractor: CitationExtractor):
        """Test extracting citations with special characters in names."""
        answer = (
            "The res.partner table [res.partner] (ID: postgresql_res_partner_123) stores partners."
        )

        citations = citation_extractor.extract_citations(answer)

        assert len(citations) == 1
        assert citations[0].artifact_name == "res.partner"

    def test_extract_duplicate_citations(self, citation_extractor: CitationExtractor):
        """Test that duplicate citations are extracted separately."""
        answer = (
            "The table [sale_order] (ID: id_123) is important. "
            "The table [sale_order] (ID: id_123) has many columns."
        )

        citations = citation_extractor.extract_citations(answer)

        # Should extract both occurrences
        assert len(citations) == 2
        assert all(c.artifact_name == "sale_order" for c in citations)


class TestCitationValidation:
    """Tests for citation validation against search results."""

    def test_validate_valid_citation(
        self,
        citation_extractor: CitationExtractor,
        sample_search_results: list[SearchResult],
    ):
        """Test validating a citation that exists in search results."""
        citations = [
            Citation(
                artifact_name="sale_order",
                artifact_id="postgresql_public_table_sale_order_123",
                is_valid=False,
                position=0,
            )
        ]

        validated = citation_extractor.validate_citations(citations, sample_search_results)

        assert len(validated) == 1
        assert validated[0].is_valid is True
        assert validated[0].matched_artifact is not None
        assert validated[0].matched_artifact.name == "sale_order"

    def test_validate_invalid_citation(
        self,
        citation_extractor: CitationExtractor,
        sample_search_results: list[SearchResult],
    ):
        """Test validating a citation that doesn't exist in search results."""
        citations = [
            Citation(
                artifact_name="unknown_table",
                artifact_id="nonexistent_id_999",
                is_valid=False,
                position=0,
            )
        ]

        validated = citation_extractor.validate_citations(citations, sample_search_results)

        assert len(validated) == 1
        assert validated[0].is_valid is False
        assert validated[0].matched_artifact is None

    def test_validate_mixed_citations(
        self,
        citation_extractor: CitationExtractor,
        sample_search_results: list[SearchResult],
    ):
        """Test validating a mix of valid and invalid citations."""
        citations = [
            Citation(
                artifact_name="sale_order",
                artifact_id="postgresql_public_table_sale_order_123",
                is_valid=False,
                position=0,
            ),
            Citation(
                artifact_name="unknown",
                artifact_id="invalid_id",
                is_valid=False,
                position=50,
            ),
            Citation(
                artifact_name="id",
                artifact_id="postgresql_public_column_sale_order_id_456",
                is_valid=False,
                position=100,
            ),
        ]

        validated = citation_extractor.validate_citations(citations, sample_search_results)

        assert len(validated) == 3
        assert validated[0].is_valid is True
        assert validated[1].is_valid is False
        assert validated[2].is_valid is True


class TestCitationExtractAndValidate:
    """Tests for combined extraction and validation."""

    def test_extract_and_validate_all_valid(
        self,
        citation_extractor: CitationExtractor,
        sample_search_results: list[SearchResult],
    ):
        """Test extract and validate with all valid citations."""
        answer = (
            "The sale_order table [sale_order] (ID: postgresql_public_table_sale_order_123) "
            "has an id column [id] (ID: postgresql_public_column_sale_order_id_456)."
        )

        citations = citation_extractor.extract_and_validate(answer, sample_search_results)

        assert len(citations) == 2
        assert all(c.is_valid for c in citations)
        assert all(c.matched_artifact is not None for c in citations)

    def test_extract_and_validate_with_hallucinations(
        self,
        citation_extractor: CitationExtractor,
        sample_search_results: list[SearchResult],
    ):
        """Test extract and validate with hallucinated citations."""
        answer = (
            "The sale_order table [sale_order] (ID: postgresql_public_table_sale_order_123) "
            "links to the customer table [customer] (ID: fake_customer_id_999)."
        )

        citations = citation_extractor.extract_and_validate(answer, sample_search_results)

        assert len(citations) == 2
        assert citations[0].is_valid is True
        assert citations[1].is_valid is False  # Hallucinated

    def test_extract_and_validate_no_citations(
        self,
        citation_extractor: CitationExtractor,
        sample_search_results: list[SearchResult],
    ):
        """Test extract and validate with no citations."""
        answer = "I don't know based on the provided context."

        citations = citation_extractor.extract_and_validate(answer, sample_search_results)

        assert len(citations) == 0


class TestCitationAnalysis:
    """Tests for citation quality analysis."""

    def test_get_cited_artifact_ids(
        self,
        citation_extractor: CitationExtractor,
        sample_search_results: list[SearchResult],
    ):
        """Test getting cited artifact IDs."""
        answer = (
            "The sale_order table [sale_order] (ID: postgresql_public_table_sale_order_123) "
            "has an id column [id] (ID: postgresql_public_column_sale_order_id_456)."
        )

        citations = citation_extractor.extract_and_validate(answer, sample_search_results)
        cited_ids = citation_extractor.get_cited_artifact_ids(citations)

        assert len(cited_ids) == 2
        assert "postgresql_public_table_sale_order_123" in cited_ids
        assert "postgresql_public_column_sale_order_id_456" in cited_ids

    def test_get_uncited_artifacts(
        self,
        citation_extractor: CitationExtractor,
        sample_search_results: list[SearchResult],
    ):
        """Test finding uncited artifacts."""
        # Only cite the first artifact
        answer = (
            "The sale_order table [sale_order] (ID: postgresql_public_table_sale_order_123) exists."
        )

        citations = citation_extractor.extract_and_validate(answer, sample_search_results)
        uncited = citation_extractor.get_uncited_artifacts(citations, sample_search_results)

        assert len(uncited) == 2  # Two artifacts not cited
        assert uncited[0].name == "id"
        assert uncited[1].name == "partner_id"

    def test_get_uncited_artifacts_all_cited(
        self,
        citation_extractor: CitationExtractor,
        sample_search_results: list[SearchResult],
    ):
        """Test when all artifacts are cited."""
        answer = (
            "The sale_order table [sale_order] (ID: postgresql_public_table_sale_order_123) "
            "has columns [id] (ID: postgresql_public_column_sale_order_id_456) "
            "and [partner_id] (ID: postgresql_public_column_sale_order_partner_id_789)."
        )

        citations = citation_extractor.extract_and_validate(answer, sample_search_results)
        uncited = citation_extractor.get_uncited_artifacts(citations, sample_search_results)

        assert len(uncited) == 0

    def test_check_citation_coverage_full(
        self,
        citation_extractor: CitationExtractor,
        sample_search_results: list[SearchResult],
    ):
        """Test citation coverage when all artifacts are cited."""
        answer = (
            "Tables: [sale_order] (ID: postgresql_public_table_sale_order_123), "
            "Columns: [id] (ID: postgresql_public_column_sale_order_id_456), "
            "[partner_id] (ID: postgresql_public_column_sale_order_partner_id_789)."
        )

        citations = citation_extractor.extract_and_validate(answer, sample_search_results)
        coverage = citation_extractor.check_citation_coverage(citations, sample_search_results)

        assert coverage == 1.0

    def test_check_citation_coverage_partial(
        self,
        citation_extractor: CitationExtractor,
        sample_search_results: list[SearchResult],
    ):
        """Test citation coverage when only some artifacts are cited."""
        answer = (
            "The sale_order table [sale_order] (ID: postgresql_public_table_sale_order_123) exists."
        )

        citations = citation_extractor.extract_and_validate(answer, sample_search_results)
        coverage = citation_extractor.check_citation_coverage(citations, sample_search_results)

        assert coverage == pytest.approx(1 / 3)  # 1 out of 3 artifacts cited

    def test_check_citation_coverage_none(
        self,
        citation_extractor: CitationExtractor,
        sample_search_results: list[SearchResult],
    ):
        """Test citation coverage when no artifacts are cited."""
        answer = "I don't know."

        citations = citation_extractor.extract_and_validate(answer, sample_search_results)
        coverage = citation_extractor.check_citation_coverage(citations, sample_search_results)

        assert coverage == 0.0

    def test_check_citation_coverage_empty_results(self, citation_extractor: CitationExtractor):
        """Test citation coverage with empty search results."""
        answer = "Some answer"
        citations = citation_extractor.extract_and_validate(answer, [])
        coverage = citation_extractor.check_citation_coverage(citations, [])

        assert coverage == 0.0

    def test_has_hallucinated_citations_true(
        self,
        citation_extractor: CitationExtractor,
        sample_search_results: list[SearchResult],
    ):
        """Test detecting hallucinated citations."""
        answer = (
            "The sale_order table [sale_order] (ID: postgresql_public_table_sale_order_123) "
            "links to [fake_table] (ID: fake_id_999)."
        )

        citations = citation_extractor.extract_and_validate(answer, sample_search_results)
        has_hallucinations = citation_extractor.has_hallucinated_citations(citations)

        assert has_hallucinations is True

    def test_has_hallucinated_citations_false(
        self,
        citation_extractor: CitationExtractor,
        sample_search_results: list[SearchResult],
    ):
        """Test when there are no hallucinated citations."""
        answer = (
            "The sale_order table [sale_order] (ID: postgresql_public_table_sale_order_123) exists."
        )

        citations = citation_extractor.extract_and_validate(answer, sample_search_results)
        has_hallucinations = citation_extractor.has_hallucinated_citations(citations)

        assert has_hallucinations is False


class TestCitationQualityMetrics:
    """Tests for citation quality metrics calculation."""

    def test_metrics_all_valid(
        self,
        citation_extractor: CitationExtractor,
        sample_search_results: list[SearchResult],
    ):
        """Test metrics when all citations are valid."""
        answer = (
            "The sale_order table [sale_order] (ID: postgresql_public_table_sale_order_123) "
            "has columns [id] (ID: postgresql_public_column_sale_order_id_456) "
            "and [partner_id] (ID: postgresql_public_column_sale_order_partner_id_789)."
        )

        metrics = citation_extractor.get_citation_quality_metrics(answer, sample_search_results)

        assert metrics["total_citations"] == 3
        assert metrics["valid_citations"] == 3
        assert metrics["invalid_citations"] == 0
        assert metrics["citation_accuracy"] == 1.0
        assert metrics["citation_coverage"] == 1.0
        assert metrics["has_hallucinations"] is False

    def test_metrics_with_hallucinations(
        self,
        citation_extractor: CitationExtractor,
        sample_search_results: list[SearchResult],
    ):
        """Test metrics with hallucinated citations."""
        answer = (
            "The sale_order table [sale_order] (ID: postgresql_public_table_sale_order_123) "
            "links to [fake1] (ID: fake_id_1) and [fake2] (ID: fake_id_2)."
        )

        metrics = citation_extractor.get_citation_quality_metrics(answer, sample_search_results)

        assert metrics["total_citations"] == 3
        assert metrics["valid_citations"] == 1
        assert metrics["invalid_citations"] == 2
        assert metrics["citation_accuracy"] == pytest.approx(1 / 3)
        assert metrics["has_hallucinations"] is True

    def test_metrics_partial_coverage(
        self,
        citation_extractor: CitationExtractor,
        sample_search_results: list[SearchResult],
    ):
        """Test metrics with partial citation coverage."""
        answer = (
            "The sale_order table [sale_order] (ID: postgresql_public_table_sale_order_123) exists."
        )

        metrics = citation_extractor.get_citation_quality_metrics(answer, sample_search_results)

        assert metrics["total_citations"] == 1
        assert metrics["valid_citations"] == 1
        assert metrics["citation_coverage"] == pytest.approx(1 / 3)

    def test_metrics_no_citations(
        self,
        citation_extractor: CitationExtractor,
        sample_search_results: list[SearchResult],
    ):
        """Test metrics with no citations."""
        answer = "I don't know based on the provided context."

        metrics = citation_extractor.get_citation_quality_metrics(answer, sample_search_results)

        assert metrics["total_citations"] == 0
        assert metrics["valid_citations"] == 0
        assert metrics["invalid_citations"] == 0
        assert metrics["citation_accuracy"] == 1.0  # No citations = perfect accuracy
        assert metrics["citation_coverage"] == 0.0
        assert metrics["has_hallucinations"] is False


class TestFormatSourcesList:
    """Tests for formatting citations as a sources list."""

    def test_format_sources_list_valid_citations(
        self,
        citation_extractor: CitationExtractor,
        sample_search_results: list[SearchResult],
    ):
        """Test formatting valid citations as a sources list."""
        answer = (
            "The sale_order table [sale_order] (ID: postgresql_public_table_sale_order_123) "
            "has an id column [id] (ID: postgresql_public_column_sale_order_id_456)."
        )

        citations = citation_extractor.extract_and_validate(answer, sample_search_results)
        sources = format_sources_list(citations)

        assert "## Sources" in sources
        assert "sale_order" in sources
        assert "id" in sources
        assert "table" in sources
        assert "column" in sources

    def test_format_sources_list_no_citations(self):
        """Test formatting empty citations list."""
        sources = format_sources_list([])

        assert "No sources cited" in sources

    def test_format_sources_list_deduplicates(
        self,
        citation_extractor: CitationExtractor,
        sample_search_results: list[SearchResult],
    ):
        """Test that duplicate citations are deduplicated in sources list."""
        answer = (
            "The table [sale_order] (ID: postgresql_public_table_sale_order_123) "
            "and the table [sale_order] (ID: postgresql_public_table_sale_order_123)."
        )

        citations = citation_extractor.extract_and_validate(answer, sample_search_results)
        sources = format_sources_list(citations)

        # The artifact name "sale_order" will appear in multiple places (name, ID),
        # but the important thing is that the artifact itself appears only once in the list
        # Count the number of list items (lines starting with "- **")
        list_items = sources.count("\n- **")
        assert list_items == 1  # Only one artifact in the sources list

    def test_format_sources_list_only_valid(
        self,
        citation_extractor: CitationExtractor,
        sample_search_results: list[SearchResult],
    ):
        """Test that only valid citations appear in sources list."""
        answer = (
            "Valid: [sale_order] (ID: postgresql_public_table_sale_order_123) "
            "Invalid: [fake] (ID: fake_id)."
        )

        citations = citation_extractor.extract_and_validate(answer, sample_search_results)
        sources = format_sources_list(citations)

        assert "sale_order" in sources
        assert "fake" not in sources
