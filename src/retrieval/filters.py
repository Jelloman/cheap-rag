"""Metadata filtering for semantic search results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class MetadataFilter:
    """Filter specification for metadata search.

    Supports filtering on language, type, tags, module, source_type, and custom metadata fields.
    Multiple filters are combined with AND logic by default.
    """

    language: str | list[str] | None = None
    type: str | list[str] | None = None
    source_type: str | list[str] | None = None
    module: str | list[str] | None = None
    tags: str | list[str] | None = None
    # Custom fields for database-specific filtering
    table_name: str | list[str] | None = None
    column_type: str | list[str] | None = None
    primary_key: bool | None = None
    # Additional custom filters
    custom: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert filter to dictionary format for ChromaDB.

        Returns:
            Dictionary of active filters with non-None values.
        """
        filters = {}

        # Standard metadata fields
        for field in ["language", "type", "source_type", "module", "table_name", "column_type"]:
            value = getattr(self, field)
            if value is not None:
                filters[field] = value

        # Boolean fields
        if self.primary_key is not None:
            filters["primary_key"] = self.primary_key

        # Tags (stored as comma-separated string in ChromaDB)
        if self.tags is not None:
            if isinstance(self.tags, list):
                # For tag filtering, we need to check if any tag matches
                # This requires special handling - for now, use first tag
                # TODO: Implement proper tag filtering with $contains in ChromaDB
                filters["tags"] = self.tags[0] if self.tags else None
            else:
                filters["tags"] = self.tags

        # Custom filters
        if self.custom:
            filters.update(self.custom)  # type: ignore[reportUnknownMemberType]

        return {k: v for k, v in filters.items() if v is not None}  # type: ignore[reportUnknownVariableType]

    def is_empty(self) -> bool:
        """Check if filter has any active constraints.

        Returns:
            True if no filters are set, False otherwise.
        """
        return len(self.to_dict()) == 0


class FilterBuilder:
    """Builder for constructing metadata filters with fluent API."""

    def __init__(self):
        """Initialize empty filter builder."""
        self._language: str | list[str] | None = None
        self._type: str | list[str] | None = None
        self._source_type: str | list[str] | None = None
        self._module: str | list[str] | None = None
        self._tags: str | list[str] | None = None
        self._table_name: str | list[str] | None = None
        self._column_type: str | list[str] | None = None
        self._primary_key: bool | None = None
        self._custom: dict[str, Any] = {}

    def language(self, language: str | list[str]) -> FilterBuilder:
        """Filter by language (e.g., "java", "postgresql", ["java", "typescript"]).

        Args:
            language: Single language or list of languages.

        Returns:
            Self for chaining.
        """
        self._language = language
        return self

    def type(self, artifact_type: str | list[str]) -> FilterBuilder:
        """Filter by artifact type (e.g., "table", "class", ["table", "view"]).

        Args:
            artifact_type: Single type or list of types.

        Returns:
            Self for chaining.
        """
        self._type = artifact_type
        return self

    def source_type(self, source_type: str | list[str]) -> FilterBuilder:
        """Filter by source type (e.g., "database", "code").

        Args:
            source_type: Single source type or list of source types.

        Returns:
            Self for chaining.
        """
        self._source_type = source_type
        return self

    def module(self, module: str | list[str]) -> FilterBuilder:
        """Filter by module/schema (e.g., "public", "com.example.app").

        Args:
            module: Single module or list of modules.

        Returns:
            Self for chaining.
        """
        self._module = module
        return self

    def tags(self, tags: str | list[str]) -> FilterBuilder:
        """Filter by tags (e.g., "core", ["database", "sales"]).

        Args:
            tags: Single tag or list of tags.

        Returns:
            Self for chaining.
        """
        self._tags = tags
        return self

    def table_name(self, table_name: str | list[str]) -> FilterBuilder:
        """Filter by table name (for column artifacts).

        Args:
            table_name: Single table name or list of table names.

        Returns:
            Self for chaining.
        """
        self._table_name = table_name
        return self

    def column_type(self, column_type: str | list[str]) -> FilterBuilder:
        """Filter by column data type (e.g., "VARCHAR", "INTEGER").

        Args:
            column_type: Single column type or list of column types.

        Returns:
            Self for chaining.
        """
        self._column_type = column_type
        return self

    def primary_key(self, is_primary_key: bool) -> FilterBuilder:
        """Filter by primary key status (for column artifacts).

        Args:
            is_primary_key: True to find only primary keys, False to exclude them.

        Returns:
            Self for chaining.
        """
        self._primary_key = is_primary_key
        return self

    def custom(self, key: str, value: Any) -> FilterBuilder:
        """Add custom metadata filter.

        Args:
            key: Metadata field name.
            value: Field value or list of values.

        Returns:
            Self for chaining.
        """
        self._custom[key] = value
        return self

    def build(self) -> MetadataFilter:
        """Build the filter object.

        Returns:
            Constructed MetadataFilter.
        """
        return MetadataFilter(
            language=self._language,
            type=self._type,
            source_type=self._source_type,
            module=self._module,
            tags=self._tags,
            table_name=self._table_name,
            column_type=self._column_type,
            primary_key=self._primary_key,
            custom=self._custom if self._custom else None,
        )


def validate_filter(filter_dict: dict[str, Any]) -> bool:
    """Validate filter dictionary has valid keys and values.

    Args:
        filter_dict: Dictionary of filter key-value pairs.

    Returns:
        True if valid, False otherwise.
    """
    valid_keys = {
        "language",
        "type",
        "source_type",
        "module",
        "tags",
        "table_name",
        "column_type",
        "primary_key",
        "nullable",
        "foreign_key",
        "unique",
        "indexed",
    }

    return all(key in valid_keys for key in filter_dict)


# Predefined filter presets for common use cases
PRESET_FILTERS = {
    "database_tables": MetadataFilter(source_type="database", type="table"),
    "database_columns": MetadataFilter(source_type="database", type="column"),
    "database_relationships": MetadataFilter(source_type="database", type="relationship"),
    "code_classes": MetadataFilter(source_type="code", type=["class", "interface"]),
    "code_methods": MetadataFilter(source_type="code", type=["method", "function"]),
    "java_artifacts": MetadataFilter(language="java"),
    "typescript_artifacts": MetadataFilter(language="typescript"),
    "python_artifacts": MetadataFilter(language="python"),
    "postgresql_artifacts": MetadataFilter(language="postgresql"),
}


def get_preset_filter(preset_name: str) -> MetadataFilter | None:
    """Get a predefined filter by name.

    Args:
        preset_name: Name of the preset filter.

    Returns:
        MetadataFilter object or None if preset not found.
    """
    return PRESET_FILTERS.get(preset_name)
