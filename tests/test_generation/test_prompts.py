"""Tests for prompt generation and formatting."""

from __future__ import annotations

import pytest

from src.extractors.base import MetadataArtifact
from src.generation.prompts import (
    build_qa_prompt,
    format_artifact_context,
    format_dont_know_response,
    format_search_results_context,
    get_citation_examples,
    get_system_message,
)
from src.retrieval.semantic_search import SearchResult


@pytest.fixture
def sample_table_artifact() -> MetadataArtifact:
    """Create a sample table artifact for testing."""
    return MetadataArtifact(
        id="postgresql_public_table_sale_order_123",
        name="sale_order",
        type="table",
        source_type="database",
        language="postgresql",
        module="public",
        description="Stores sales orders for customers",
        tags=["sales", "orders"],
        relations=["sale_order_line", "res_partner"],
        metadata={},
    )


@pytest.fixture
def sample_column_artifact() -> MetadataArtifact:
    """Create a sample column artifact for testing."""
    return MetadataArtifact(
        id="postgresql_public_column_sale_order_id_456",
        name="id",
        type="column",
        source_type="database",
        language="postgresql",
        module="public",
        description="Primary key for sale_order table",
        metadata={
            "table_name": "sale_order",
            "column_type": "integer",
            "nullable": False,
            "primary_key": True,
            "unique": True,
            "indexed": True,
        },
    )


@pytest.fixture
def sample_search_results(
    sample_table_artifact: MetadataArtifact,
    sample_column_artifact: MetadataArtifact,
) -> list[SearchResult]:
    """Create sample search results."""
    return [
        SearchResult(
            artifact=sample_table_artifact,
            similarity=0.85,
            distance=0.15,
            rank=1,
        ),
        SearchResult(
            artifact=sample_column_artifact,
            similarity=0.72,
            distance=0.28,
            rank=2,
        ),
    ]


class TestSystemMessages:
    """Tests for system message generation."""

    def test_get_system_message_ollama(self):
        """Test getting Ollama system message."""
        message = get_system_message("ollama")
        assert isinstance(message, str)
        assert len(message) > 0
        assert "Qwen" in message
        assert "provided context" in message.lower()
        assert "cite" in message.lower()

    def test_get_system_message_anthropic(self):
        """Test getting Anthropic system message."""
        message = get_system_message("anthropic")
        assert isinstance(message, str)
        assert len(message) > 0
        assert "Claude" in message
        assert "provided context" in message.lower()
        assert "cite" in message.lower()

    def test_get_system_message_hybrid(self):
        """Test getting hybrid/base system message."""
        message = get_system_message("hybrid")
        assert isinstance(message, str)
        assert len(message) > 0
        assert "provided context" in message.lower()

    def test_get_system_message_default(self):
        """Test getting default system message."""
        message = get_system_message()
        assert isinstance(message, str)
        assert "Qwen" in message  # Defaults to ollama

    def test_all_system_messages_have_citation_instructions(self):
        """Test that all system messages include citation instructions."""
        providers = ["ollama", "anthropic", "hybrid"]
        for provider in providers:
            message = get_system_message(provider)
            # Check for citation format
            assert "[" in message and "ID:" in message


class TestArtifactFormatting:
    """Tests for artifact context formatting."""

    def test_format_table_artifact(self, sample_table_artifact: MetadataArtifact):
        """Test formatting a table artifact."""
        context = format_artifact_context(sample_table_artifact)

        assert "sale_order" in context
        assert "TABLE:" in context.upper()
        assert sample_table_artifact.id in context
        assert "postgresql" in context
        assert "Stores sales orders" in context
        assert "Related Tables" in context
        assert "sale_order_line" in context

    def test_format_column_artifact(self, sample_column_artifact: MetadataArtifact):
        """Test formatting a column artifact."""
        context = format_artifact_context(sample_column_artifact)

        assert "id" in context
        assert "COLUMN:" in context.upper()
        assert sample_column_artifact.id in context
        assert "sale_order" in context
        assert "integer" in context
        assert "**Primary Key:** Yes" in context
        assert "**Nullable:** No" in context

    def test_format_artifact_with_source_file(self):
        """Test formatting an artifact with source file information."""
        artifact = MetadataArtifact(
            id="java_class_Catalog_789",
            name="Catalog",
            type="class",
            source_type="code",
            language="java",
            module="com.example.cheap",
            description="Main catalog interface",
            source_file="Catalog.java",
            source_line=10,
            metadata={},
        )

        context = format_artifact_context(artifact)

        assert "Catalog.java:10" in context
        assert "Source:" in context

    def test_format_artifact_with_tags(self, sample_table_artifact: MetadataArtifact):
        """Test formatting an artifact with tags."""
        context = format_artifact_context(sample_table_artifact)

        assert "Tags:" in context
        assert "sales" in context
        assert "orders" in context

    def test_format_relationship_artifact(self):
        """Test formatting a relationship artifact."""
        artifact = MetadataArtifact(
            id="postgresql_public_relationship_sale_order_partner_123",
            name="sale_order_partner_fkey",
            type="relationship",
            source_type="database",
            language="postgresql",
            module="public",
            description="Links orders to partners",
            metadata={
                "from_table": "sale_order",
                "to_table": "res_partner",
                "cardinality": "many-to-one",
            },
        )

        context = format_artifact_context(artifact)

        assert "relationship" in context.lower()
        assert "**From Table:** sale_order" in context
        assert "**To Table:** res_partner" in context
        assert "**Cardinality:** many-to-one" in context

    def test_format_index_artifact(self):
        """Test formatting an index artifact."""
        artifact = MetadataArtifact(
            id="postgresql_public_index_sale_order_idx_123",
            name="sale_order_date_idx",
            type="index",
            source_type="database",
            language="postgresql",
            module="public",
            description="Index on order date",
            metadata={
                "table_name": "sale_order",
                "unique": True,
                "columns": ["date_order", "id"],
            },
        )

        context = format_artifact_context(artifact)

        assert "index" in context.lower()
        assert "**Table:** sale_order" in context
        assert "**Type:** UNIQUE" in context
        assert "**Columns:** date_order, id" in context


class TestSearchResultsFormatting:
    """Tests for search results context formatting."""

    def test_format_search_results_with_results(self, sample_search_results: list[SearchResult]):
        """Test formatting search results with multiple artifacts."""
        context = format_search_results_context(sample_search_results)

        assert "Retrieved Metadata Artifacts" in context
        assert "Result 1" in context
        assert "Result 2" in context
        assert "0.850" in context  # similarity score
        assert "0.720" in context
        assert "sale_order" in context
        assert "id" in context

    def test_format_search_results_empty(self):
        """Test formatting empty search results."""
        context = format_search_results_context([])

        assert "No relevant artifacts" in context

    def test_format_search_results_single_result(self, sample_table_artifact: MetadataArtifact):
        """Test formatting search results with a single artifact."""
        results = [
            SearchResult(
                artifact=sample_table_artifact,
                similarity=0.95,
                distance=0.05,
                rank=1,
            )
        ]

        context = format_search_results_context(results)

        assert "Result 1" in context
        assert "0.950" in context
        assert "sale_order" in context
        # Should not have "Result 2"
        assert "Result 2" not in context


class TestQAPromptBuilding:
    """Tests for complete Q&A prompt building."""

    def test_build_qa_prompt_with_results(self, sample_search_results: list[SearchResult]):
        """Test building a complete Q&A prompt with search results."""
        query = "What is the sale_order table used for?"

        prompt = build_qa_prompt(query, sample_search_results)

        # Check structure
        assert "Retrieved Metadata Artifacts" in prompt
        assert "User Question" in prompt
        assert "Instructions" in prompt

        # Check content
        assert query in prompt
        assert "sale_order" in prompt
        assert "cite your sources" in prompt.lower()
        assert "[ArtifactName] (ID: artifact_id)" in prompt

        # Check it includes search results
        assert "0.850" in prompt
        assert "0.720" in prompt

    def test_build_qa_prompt_empty_results(self):
        """Test building a prompt with no search results."""
        query = "What is the unknown_table used for?"

        prompt = build_qa_prompt(query, [])

        assert query in prompt
        assert "No relevant artifacts" in prompt
        assert "User Question" in prompt

    def test_build_qa_prompt_includes_citation_format(
        self, sample_search_results: list[SearchResult]
    ):
        """Test that the prompt includes citation format instructions."""
        query = "Test query"
        prompt = build_qa_prompt(query, sample_search_results)

        assert "[ArtifactName]" in prompt
        assert "(ID: artifact_id)" in prompt


class TestCitationHelpers:
    """Tests for citation helper functions."""

    def test_get_citation_examples(self):
        """Test getting citation examples."""
        examples = get_citation_examples()

        assert isinstance(examples, str)
        assert "Example Citations" in examples
        assert "[sale_order]" in examples
        assert "(ID:" in examples

    def test_format_dont_know_response(self):
        """Test formatting 'don't know' response."""
        response = format_dont_know_response()

        assert isinstance(response, str)
        assert "don't know" in response.lower()
        assert "provided context" in response.lower()


class TestPromptConsistency:
    """Tests for consistency across prompt components."""

    def test_citation_format_consistency(self, sample_search_results: list[SearchResult]):
        """Test that citation format is consistent across all prompts."""
        citation_format = "[ArtifactName] (ID: artifact_id)"

        # Check system messages
        for provider in ["ollama", "anthropic", "hybrid"]:
            system_message = get_system_message(provider)
            assert citation_format in system_message or "[" in system_message

        # Check Q&A prompt
        qa_prompt = build_qa_prompt("test", sample_search_results)
        assert citation_format in qa_prompt

        # Check citation examples
        examples = get_citation_examples()
        assert "(ID:" in examples

    def test_dont_know_instruction_consistency(self):
        """Test that 'don't know' instructions are consistent."""
        dont_know_phrase = "I don't know based on the provided context"

        # Should be in system messages
        for provider in ["ollama", "anthropic", "hybrid"]:
            system_message = get_system_message(provider)
            assert "don't know" in system_message.lower()

        # Should be in Q&A prompt
        qa_prompt = build_qa_prompt("test", [])
        assert "don't know" in qa_prompt.lower()

        # Should match the standard response
        response = format_dont_know_response()
        assert dont_know_phrase.lower() in response.lower()
