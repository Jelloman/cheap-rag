"""JSON Schema for metadata artifacts."""

from __future__ import annotations

from typing import Any

from src.extractors.base import MetadataArtifact

# JSON Schema for MetadataArtifact validation
METADATA_ARTIFACT_SCHEMA: dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "MetadataArtifact",
    "description": "Normalized metadata artifact from databases and code",
    "type": "object",
    "required": ["id", "name", "type", "source_type", "language", "module", "description"],
    "properties": {
        "id": {
            "type": "string",
            "description": "Unique identifier (hash of qualified name + language)",
        },
        "name": {
            "type": "string",
            "description": "Table/column/class/interface/method name",
        },
        "type": {
            "type": "string",
            "enum": [
                # Database types
                "table",
                "column",
                "index",
                "constraint",
                "relationship",
                "view",
                "trigger",
                # Code types
                "class",
                "interface",
                "field",
                "method",
                "function",
                "type",
            ],
            "description": "Type of metadata artifact",
        },
        "source_type": {
            "type": "string",
            "enum": ["database", "code", "csv", "key_value"],
            "description": "Source type of the artifact",
        },
        "language": {
            "type": "string",
            "enum": [
                # Database languages
                "postgresql",
                "sqlite",
                "mariadb",
                "mysql",
                # Code languages
                "java",
                "typescript",
                "python",
                "rust",
                # File formats
                "csv",
                "json",
                "parquet",
            ],
            "description": "Source language or technology",
        },
        "module": {
            "type": "string",
            "description": "Package/module/namespace path",
        },
        "description": {
            "type": "string",
            "description": "Documentation or comment text",
        },
        "constraints": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Validation rules and constraints",
            "default": [],
        },
        "relations": {
            "type": "array",
            "items": {"type": "string"},
            "description": "References to other artifacts",
            "default": [],
        },
        "examples": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Usage examples",
            "default": [],
        },
        "tags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Categorization tags",
            "default": [],
        },
        "source_file": {
            "type": "string",
            "description": "Path to source file",
            "default": "",
        },
        "source_line": {
            "type": "integer",
            "description": "Line number in source file",
            "default": 0,
        },
        "metadata": {
            "type": "object",
            "description": "Extensible metadata dictionary",
            "default": {},
        },
    },
}


class MetadataSchema:
    """Metadata schema utilities."""

    @staticmethod
    def get_schema() -> dict[str, Any]:
        """Get the JSON schema for metadata artifacts."""
        return METADATA_ARTIFACT_SCHEMA

    @staticmethod
    def validate(artifact_dict: dict[str, Any]) -> bool:
        """Validate a metadata artifact against the schema.

        Args:
            artifact_dict: Dictionary representation of artifact.

        Returns:
            True if valid, raises exception otherwise.
        """
        # Import jsonschema only when needed
        try:
            import jsonschema

            jsonschema.validate(instance=artifact_dict, schema=METADATA_ARTIFACT_SCHEMA)
            return True
        except ImportError:
            # If jsonschema not installed, just check required fields
            required = ["id", "name", "type", "source_type", "language", "module", "description"]
            for field in required:
                if field not in artifact_dict:
                    raise ValueError(f"Missing required field: {field}") from None
            return True


def validate_artifact(artifact: MetadataArtifact) -> tuple[bool, list[str]]:
    """Validate a metadata artifact against the schema.

    Args:
        artifact: Metadata artifact to validate.

    Returns:
        Tuple of (is_valid, error_messages).
    """
    errors = []

    # Check required fields
    required_fields = ["id", "name", "type", "source_type", "language", "module", "description"]
    for field in required_fields:
        value = getattr(artifact, field, None)
        if not value:
            errors.append(f"Missing or empty required field: {field}")  # type: ignore[reportUnknownMemberType]

    # Check type enums
    valid_types = [
        "table",
        "column",
        "index",
        "constraint",
        "relationship",
        "view",
        "trigger",
        "class",
        "interface",
        "field",
        "method",
        "function",
        "type",
    ]
    if artifact.type and artifact.type not in valid_types:
        errors.append(f"Invalid artifact type: {artifact.type}")  # type: ignore[reportUnknownMemberType]

    valid_source_types = ["database", "code", "csv", "key_value"]
    if artifact.source_type and artifact.source_type not in valid_source_types:
        errors.append(f"Invalid source_type: {artifact.source_type}")  # type: ignore[reportUnknownMemberType]

    valid_languages = [
        "postgresql",
        "sqlite",
        "mariadb",
        "mysql",
        "java",
        "typescript",
        "python",
        "rust",
        "csv",
        "json",
        "parquet",
    ]
    if artifact.language and artifact.language not in valid_languages:
        errors.append(f"Invalid language: {artifact.language}")  # type: ignore[reportUnknownMemberType]

    return len(errors) == 0, errors  # type: ignore[reportUnknownVariableType,reportUnknownArgumentType]
