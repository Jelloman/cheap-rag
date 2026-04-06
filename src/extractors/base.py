"""Base classes for metadata extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass(slots=True)
class MetadataArtifact:
    """Normalized metadata artifact from databases and code.

    This is the core data structure that represents extracted metadata
    from databases (PostgreSQL, SQLite, MariaDB) and code (Java, TypeScript, Python, Rust).

    Database artifacts (tables, columns, indexes, relationships) and code artifacts
    (classes, interfaces, methods) share a unified model with type-specific metadata.
    """

    # Core identification
    id: str  # Unique identifier (hash of qualified name + language)
    name: str  # Table/column/class/interface/method name
    type: str  # "table", "column", "index", "constraint", "relationship", "view",
    # "class", "interface", "field", "method", "function"
    source_type: str  # "database", "code", "csv", "key_value"

    # Language/technology identifier
    # For databases: "postgresql", "sqlite", "mariadb", "mysql"
    # For code: "java", "typescript", "python", "rust"
    # For files: "csv", "json", "parquet"
    language: str

    # Organizational
    module: str  # For databases: schema name; for code: package/module/namespace
    description: str  # Documentation, comment text, or table comment

    # Metadata details
    constraints: list[str] = field(default_factory=list)  # Validation rules, check constraints
    relations: list[str] = field(
        default_factory=list
    )  # Foreign keys, references to other artifacts
    examples: list[str] = field(default_factory=list)  # Usage examples
    tags: list[str] = field(
        default_factory=list
    )  # Categorization (e.g., "database", "table", "core")

    # Source location (for code; empty for databases)
    source_file: str = ""  # Path to original definition
    source_line: int = 0  # Line number for navigation

    # Extensible type-specific metadata
    # For database columns: "column_type", "nullable", "default_value", "primary_key",
    #                       "foreign_key", "unique", "indexed", "table_name"
    # For database tables: "row_count", "schema", "table_type"
    # For relationships: "from_table", "to_table", "from_columns", "to_columns", "cardinality"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "source_type": self.source_type,
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
    def from_dict(cls, data: dict[str, Any]) -> MetadataArtifact:
        """Create from dictionary."""
        return cls(**data)

    def to_embedding_text(self) -> str:
        """Convert to text suitable for embedding.

        Creates semantic text representation optimized for different artifact types.
        Database artifacts emphasize schema and relationships; code artifacts
        emphasize structure and documentation.
        """
        # Database table
        if self.type == "table":
            parts = [
                f"[TABLE] {self.name} in {self.language} database",
                f"Schema: {self.module}",
            ]
            if self.description:
                parts.append(f"Description: {self.description}")
            if self.tags:
                parts.append(f"Tags: {', '.join(self.tags)}")
            if self.relations:
                parts.append(f"Related tables: {', '.join(self.relations)}")
            return "\n".join(parts)

        # Database column
        elif self.type == "column":
            table_name = self.metadata.get("table_name", "")
            col_type = self.metadata.get("column_type", "")
            nullable = self.metadata.get("nullable", True)
            pk = self.metadata.get("primary_key", False)
            fk = self.metadata.get("foreign_key", "")

            parts = [
                f"[COLUMN] {self.name} in table {table_name}",
                f"Type: {col_type}",
                f"Nullable: {'Yes' if nullable else 'No'}",
            ]
            if pk:
                parts.append("Primary Key: Yes")
            if fk:
                parts.append(f"Foreign Key to: {fk}")
            if self.description:
                parts.append(f"Description: {self.description}")
            if self.constraints:
                parts.append(f"Constraints: {', '.join(self.constraints)}")
            return "\n".join(parts)

        # Database relationship (foreign key)
        elif self.type == "relationship":
            from_table = self.metadata.get("from_table", "")
            to_table = self.metadata.get("to_table", "")
            cardinality = self.metadata.get("cardinality", "")

            parts = [
                f"[RELATIONSHIP] Foreign key from {from_table} to {to_table}",
                f"Name: {self.name}",
            ]
            if cardinality:
                parts.append(f"Cardinality: {cardinality}")
            if self.description:
                parts.append(f"Description: {self.description}")
            if self.relations:
                parts.append(f"Columns: {', '.join(self.relations)}")
            return "\n".join(parts)

        # Database index
        elif self.type == "index":
            table_name = self.metadata.get("table_name", "")
            unique = self.metadata.get("unique", False)
            columns = self.metadata.get("columns", [])

            parts = [
                f"[INDEX] {self.name} on table {table_name}",
                f"Type: {'UNIQUE' if unique else 'NON-UNIQUE'}",
            ]
            if columns:
                # Filter out None values before joining
                column_names = [str(c) for c in columns if c is not None]
                if column_names:
                    parts.append(f"Columns: {', '.join(column_names)}")
            if self.description:
                parts.append(f"Description: {self.description}")
            return "\n".join(parts)

        # Code artifacts (classes, interfaces, methods, etc.)
        else:
            # Use pre-computed embedding text if provided (e.g. by Java extractor JAR)
            if self.metadata.get("embedding_text"):
                return str(self.metadata["embedding_text"])

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


@runtime_checkable
class MetadataExtractor(Protocol):
    """Protocol for language-specific metadata extractors."""

    def extract_metadata(self, source_path: Path) -> list[MetadataArtifact]:
        """Extract metadata from source files.

        Args:
            source_path: Path to source file or directory.

        Returns:
            List of extracted metadata artifacts.
        """
        ...

    def language(self) -> str:
        """Return the language identifier for this extractor."""
        ...
