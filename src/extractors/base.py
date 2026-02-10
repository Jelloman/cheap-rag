"""Base classes for metadata extraction."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class MetadataArtifact:
    """Normalized metadata artifact across all languages.

    This is the core data structure that represents extracted metadata
    from any supported language (Java, TypeScript, Python, Rust).
    """

    # Core identification
    id: str  # Unique identifier (hash of qualified name + language)
    name: str  # Class/interface/type/field/method name
    type: str  # "class", "interface", "field", "method", "constraint", etc.
    language: str  # "java", "typescript", "python", "rust"

    # Organizational
    module: str  # Package/module/namespace
    description: str  # Docstring or comment text

    # Metadata details
    constraints: list[str] = field(default_factory=list)  # Validation rules
    relations: list[str] = field(default_factory=list)  # References to other artifacts
    examples: list[str] = field(default_factory=list)  # Usage examples
    tags: list[str] = field(default_factory=list)  # Categorization

    # Source location
    source_file: str = ""  # Path to original definition
    source_line: int = 0  # Line number for navigation

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)  # Extensible metadata

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "language": self.language,
            "module": self.module,
            "description": self.description,
            "constraints": self.constraints,
            "relations": self.relations,
            "examples": self.examples,
            "tags": self.tags,
            "source_file": self.source_file,
            "source_line": self.source_line,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MetadataArtifact":
        """Create from dictionary."""
        return cls(**data)

    def to_embedding_text(self) -> str:
        """Convert to text suitable for embedding.

        This creates a structured text representation that includes
        the most important information for semantic search.
        """
        parts = [
            f"[{self.type.upper()}] {self.name} in {self.language}",
            f"Module: {self.module}",
        ]

        if self.description:
            parts.append(f"Description: {self.description}")

        if self.constraints:
            parts.append(f"Constraints: {', '.join(self.constraints)}")

        if self.tags:
            parts.append(f"Tags: {', '.join(self.tags)}")

        if self.relations:
            parts.append(f"Related to: {', '.join(self.relations)}")

        return "\n".join(parts)


class MetadataExtractor(ABC):
    """Base class for language-specific metadata extractors."""

    @abstractmethod
    def extract_metadata(self, source_path: Path) -> list[MetadataArtifact]:
        """Extract metadata from source files.

        Args:
            source_path: Path to source file or directory.

        Returns:
            List of extracted metadata artifacts.
        """
        pass

    @abstractmethod
    def language(self) -> str:
        """Return the language identifier for this extractor."""
        pass
