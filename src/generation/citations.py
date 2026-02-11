"""Citation extraction and validation from LLM-generated answers."""

import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Set

from src.extractors.base import MetadataArtifact
from src.retrieval.semantic_search import SearchResult

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    """Parsed citation from answer text."""

    artifact_name: str
    artifact_id: str
    is_valid: bool
    position: int  # Character position in answer
    matched_artifact: Optional[MetadataArtifact] = None

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            "artifact_name": self.artifact_name,
            "artifact_id": self.artifact_id,
            "is_valid": self.is_valid,
            "position": self.position,
            "matched_artifact": self.matched_artifact.to_dict() if self.matched_artifact else None,
        }


class CitationExtractor:
    """Extracts and validates citations from LLM-generated answers."""

    # Citation pattern: [ArtifactName] (ID: artifact_id)
    CITATION_PATTERN = r'\[([^\]]+)\]\s*\(ID:\s*([^\)]+)\)'

    def __init__(self):
        """Initialize citation extractor."""
        self.pattern = re.compile(self.CITATION_PATTERN)

    def extract_citations(self, answer: str) -> List[Citation]:
        """Extract all citations from answer text.

        Args:
            answer: Generated answer text.

        Returns:
            List of parsed citations.
        """
        citations = []

        for match in self.pattern.finditer(answer):
            artifact_name = match.group(1).strip()
            artifact_id = match.group(2).strip()
            position = match.start()

            citation = Citation(
                artifact_name=artifact_name,
                artifact_id=artifact_id,
                is_valid=False,  # Will be validated separately
                position=position,
            )
            citations.append(citation)

        logger.debug(f"Extracted {len(citations)} citations from answer")
        return citations

    def validate_citations(
        self,
        citations: List[Citation],
        search_results: List[SearchResult],
    ) -> List[Citation]:
        """Validate that citations reference artifacts from search results.

        Args:
            citations: List of extracted citations.
            search_results: Retrieved search results.

        Returns:
            List of citations with validation status and matched artifacts.
        """
        # Build ID to artifact mapping
        artifact_map = {r.artifact.id: r.artifact for r in search_results}

        validated_citations = []
        for citation in citations:
            # Check if artifact ID exists in results
            if citation.artifact_id in artifact_map:
                citation.is_valid = True
                citation.matched_artifact = artifact_map[citation.artifact_id]
                logger.debug(f"Valid citation: {citation.artifact_name} -> {citation.artifact_id}")
            else:
                citation.is_valid = False
                logger.warning(
                    f"Invalid citation: {citation.artifact_name} (ID: {citation.artifact_id}) "
                    "not found in search results"
                )

            validated_citations.append(citation)

        valid_count = sum(1 for c in validated_citations if c.is_valid)
        logger.info(f"Validated {len(citations)} citations: {valid_count} valid, "
                   f"{len(citations) - valid_count} invalid")

        return validated_citations

    def extract_and_validate(
        self,
        answer: str,
        search_results: List[SearchResult],
    ) -> List[Citation]:
        """Extract and validate citations in one step.

        Args:
            answer: Generated answer text.
            search_results: Retrieved search results.

        Returns:
            List of validated citations.
        """
        citations = self.extract_citations(answer)
        return self.validate_citations(citations, search_results)

    def get_cited_artifact_ids(self, citations: List[Citation]) -> Set[str]:
        """Get set of all artifact IDs that were cited.

        Args:
            citations: List of citations.

        Returns:
            Set of artifact IDs.
        """
        return {c.artifact_id for c in citations if c.is_valid}

    def get_uncited_artifacts(
        self,
        citations: List[Citation],
        search_results: List[SearchResult],
    ) -> List[MetadataArtifact]:
        """Get artifacts that were retrieved but not cited.

        Args:
            citations: List of validated citations.
            search_results: Retrieved search results.

        Returns:
            List of uncited artifacts.
        """
        cited_ids = self.get_cited_artifact_ids(citations)
        uncited = []

        for result in search_results:
            if result.artifact.id not in cited_ids:
                uncited.append(result.artifact)

        if uncited:
            logger.info(f"Found {len(uncited)} uncited artifacts out of "
                       f"{len(search_results)} retrieved")

        return uncited

    def check_citation_coverage(
        self,
        citations: List[Citation],
        search_results: List[SearchResult],
    ) -> float:
        """Calculate percentage of retrieved artifacts that were cited.

        Args:
            citations: List of citations.
            search_results: Retrieved search results.

        Returns:
            Coverage ratio (0.0 to 1.0).
        """
        if not search_results:
            return 0.0

        cited_ids = self.get_cited_artifact_ids(citations)
        coverage = len(cited_ids) / len(search_results)

        logger.debug(f"Citation coverage: {coverage:.2%} "
                    f"({len(cited_ids)}/{len(search_results)} artifacts)")

        return coverage

    def has_hallucinated_citations(self, citations: List[Citation]) -> bool:
        """Check if answer contains invalid citations (hallucinations).

        Args:
            citations: List of validated citations.

        Returns:
            True if any citations are invalid.
        """
        invalid_count = sum(1 for c in citations if not c.is_valid)

        if invalid_count > 0:
            logger.warning(f"Found {invalid_count} hallucinated citations")
            return True

        return False

    def get_citation_quality_metrics(
        self,
        answer: str,
        search_results: List[SearchResult],
    ) -> dict:
        """Calculate citation quality metrics.

        Args:
            answer: Generated answer text.
            search_results: Retrieved search results.

        Returns:
            Dictionary of metrics.
        """
        citations = self.extract_and_validate(answer, search_results)

        total_citations = len(citations)
        valid_citations = sum(1 for c in citations if c.is_valid)
        invalid_citations = total_citations - valid_citations
        coverage = self.check_citation_coverage(citations, search_results)
        has_hallucinations = self.has_hallucinated_citations(citations)

        metrics = {
            "total_citations": total_citations,
            "valid_citations": valid_citations,
            "invalid_citations": invalid_citations,
            "citation_accuracy": valid_citations / total_citations if total_citations > 0 else 1.0,
            "citation_coverage": coverage,
            "has_hallucinations": has_hallucinations,
        }

        logger.info(f"Citation quality: {metrics['citation_accuracy']:.2%} accuracy, "
                   f"{metrics['citation_coverage']:.2%} coverage")

        return metrics


def format_sources_list(citations: List[Citation]) -> str:
    """Format citations as a sources list.

    Args:
        citations: List of validated citations.

    Returns:
        Formatted sources string.
    """
    if not citations:
        return "No sources cited."

    lines = ["## Sources"]
    seen_ids = set()

    for citation in citations:
        if citation.is_valid and citation.artifact_id not in seen_ids:
            artifact = citation.matched_artifact
            if artifact:
                source_info = f"- **{artifact.name}** ({artifact.type})"
                source_info += f" - {artifact.language}"
                if artifact.module:
                    source_info += f" - {artifact.module}"
                source_info += f"\n  ID: {artifact.id}"

                if artifact.description:
                    desc = artifact.description[:100]
                    if len(artifact.description) > 100:
                        desc += "..."
                    source_info += f"\n  {desc}"

                lines.append(source_info)
                seen_ids.add(citation.artifact_id)

    return "\n".join(lines)
