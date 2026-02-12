"""Structured response formatting for query answers."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_serializer

from src.extractors.base import MetadataArtifact
from src.generation.citations import Citation
from src.retrieval.semantic_search import SearchResult


class ArtifactSummary(BaseModel):
    """Summary of a metadata artifact for response."""

    id: str
    name: str
    type: str
    language: str
    module: str
    description: str
    similarity: float | None = None
    rank: int | None = None

    @classmethod
    def from_artifact(
        cls,
        artifact: MetadataArtifact,
        similarity: float | None = None,
        rank: int | None = None,
    ) -> ArtifactSummary:
        """Create summary from MetadataArtifact.

        Args:
            artifact: Full metadata artifact.
            similarity: Similarity score.
            rank: Search result rank.

        Returns:
            ArtifactSummary instance.
        """
        return cls(
            id=artifact.id,
            name=artifact.name,
            type=artifact.type,
            language=artifact.language,
            module=artifact.module,
            description=artifact.description[:200] if artifact.description else "",
            similarity=similarity,
            rank=rank,
        )

    @classmethod
    def from_search_result(cls, result: SearchResult) -> ArtifactSummary:
        """Create summary from SearchResult.

        Args:
            result: Search result.

        Returns:
            ArtifactSummary instance.
        """
        return cls.from_artifact(
            artifact=result.artifact,
            similarity=result.similarity,
            rank=result.rank,
        )


class CitationInfo(BaseModel):
    """Information about a citation in the answer."""

    artifact_name: str
    artifact_id: str
    is_valid: bool
    artifact_summary: ArtifactSummary | None = None

    @classmethod
    def from_citation(cls, citation: Citation) -> CitationInfo:
        """Create from Citation object.

        Args:
            citation: Parsed citation.

        Returns:
            CitationInfo instance.
        """
        summary = None
        if citation.matched_artifact:
            summary = ArtifactSummary.from_artifact(citation.matched_artifact)

        return cls(
            artifact_name=citation.artifact_name,
            artifact_id=citation.artifact_id,
            is_valid=citation.is_valid,
            artifact_summary=summary,
        )


class SearchMetadata(BaseModel):
    """Metadata about the search that produced this answer."""

    query: str
    top_k: int
    similarity_threshold: float
    num_results: int
    filters: dict[str, Any] | None = None
    retrieval_time_ms: float | None = None


class GenerationMetadata(BaseModel):
    """Metadata about the LLM generation process."""

    provider: str
    model: str
    temperature: float
    max_tokens: int
    generation_time_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class CitationMetrics(BaseModel):
    """Quality metrics for citations in the answer."""

    total_citations: int = 0
    valid_citations: int = 0
    invalid_citations: int = 0
    citation_accuracy: float = 1.0
    citation_coverage: float = 0.0
    has_hallucinations: bool = False


class QueryResponse(BaseModel):
    """Complete response to a user query."""

    # Answer content
    answer: str
    query: str

    # Sources and citations
    citations: list[CitationInfo] = Field(default_factory=list)
    sources: list[ArtifactSummary] = Field(default_factory=list)

    # Metadata
    search_metadata: SearchMetadata
    generation_metadata: GenerationMetadata | None = None
    citation_metrics: CitationMetrics | None = None

    # Timestamps
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    total_time_ms: float | None = None

    # Quality indicators
    confidence: str | None = None  # "high", "medium", "low"
    warnings: list[str] = Field(default_factory=list)

    model_config = ConfigDict(ser_json_timedelta="iso8601")

    @field_serializer("timestamp")
    def serialize_timestamp(self, dt: datetime) -> str:
        """Serialize datetime to ISO format."""
        return dt.isoformat()

    def add_warning(self, warning: str) -> None:
        """Add a warning to the response.

        Args:
            warning: Warning message.
        """
        self.warnings.append(warning)

    def assess_confidence(self) -> str:
        """Assess answer confidence based on metrics.

        Returns:
            Confidence level: "high", "medium", or "low".
        """
        # Start with medium confidence
        confidence = "medium"

        # Check search results quality
        if self.sources:
            avg_similarity = sum(s.similarity for s in self.sources if s.similarity) / len(
                self.sources
            )

            if avg_similarity > 0.7:
                confidence = "high"
            elif avg_similarity < 0.4:
                confidence = "low"

        # Downgrade if citation issues
        if self.citation_metrics:
            if self.citation_metrics.has_hallucinations:
                confidence = "low"
            elif self.citation_metrics.citation_accuracy < 0.8 and confidence == "high":
                confidence = "medium"

        # Downgrade if very few results
        if self.search_metadata.num_results < 2 and confidence == "high":
            confidence = "medium"

        # Check for "don't know" responses
        if "don't know" in self.answer.lower() or "insufficient" in self.answer.lower():
            confidence = "low"

        self.confidence = confidence
        return confidence

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation.
        """
        return self.model_dump()

    def to_markdown(self) -> str:
        """Format response as markdown for display.

        Returns:
            Markdown-formatted response.
        """
        lines = []

        # Query
        lines.append(f"# Query: {self.query}")
        lines.append("")

        # Answer
        lines.append("## Answer")
        lines.append(self.answer)
        lines.append("")

        # Sources
        if self.sources:
            lines.append("## Sources")
            for i, source in enumerate(self.sources, 1):
                lines.append(f"{i}. **{source.name}** ({source.type})")
                lines.append(f"   - Language: {source.language}")
                lines.append(f"   - Module: {source.module}")
                if source.similarity:
                    lines.append(f"   - Similarity: {source.similarity:.3f}")
                if source.description:
                    lines.append(f"   - {source.description}")
                lines.append("")

        # Metadata
        lines.append("## Metadata")
        lines.append(f"- **Query Time:** {self.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        if self.total_time_ms:
            lines.append(f"- **Total Time:** {self.total_time_ms:.0f}ms")
        if self.confidence:
            lines.append(f"- **Confidence:** {self.confidence}")
        lines.append(f"- **Retrieved Artifacts:** {self.search_metadata.num_results}")
        if self.generation_metadata:
            lines.append(f"- **LLM Provider:** {self.generation_metadata.provider}")
        lines.append("")

        # Citation metrics
        if self.citation_metrics and self.citation_metrics.total_citations > 0:
            lines.append("## Citation Quality")
            lines.append(f"- **Total Citations:** {self.citation_metrics.total_citations}")
            lines.append(f"- **Valid Citations:** {self.citation_metrics.valid_citations}")
            lines.append(f"- **Accuracy:** {self.citation_metrics.citation_accuracy:.1%}")
            lines.append(f"- **Coverage:** {self.citation_metrics.citation_coverage:.1%}")
            lines.append("")

        # Warnings
        if self.warnings:
            lines.append("## Warnings")
            for warning in self.warnings:
                lines.append(f"- ⚠️ {warning}")
            lines.append("")

        return "\n".join(lines)


class ErrorResponse(BaseModel):
    """Error response for failed queries."""

    error: str
    error_type: str
    query: str | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    details: dict[str, Any] | None = None

    model_config = ConfigDict(ser_json_timedelta="iso8601")

    @field_serializer("timestamp")
    def serialize_timestamp(self, dt: datetime) -> str:
        """Serialize datetime to ISO format."""
        return dt.isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return self.model_dump()
