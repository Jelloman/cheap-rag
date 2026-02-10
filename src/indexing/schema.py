"""JSON Schema for metadata artifacts."""

from typing import Any

# JSON Schema for MetadataArtifact validation
METADATA_ARTIFACT_SCHEMA: dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "MetadataArtifact",
    "description": "Normalized metadata artifact from CHEAP schema definitions",
    "type": "object",
    "required": ["id", "name", "type", "language", "module", "description"],
    "properties": {
        "id": {
            "type": "string",
            "description": "Unique identifier (hash of qualified name + language)",
        },
        "name": {
            "type": "string",
            "description": "Class/interface/type/field/method name",
        },
        "type": {
            "type": "string",
            "enum": ["class", "interface", "field", "method", "constraint", "type", "function"],
            "description": "Type of metadata artifact",
        },
        "language": {
            "type": "string",
            "enum": ["java", "typescript", "python", "rust"],
            "description": "Source language",
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
            required = ["id", "name", "type", "language", "module", "description"]
            for field in required:
                if field not in artifact_dict:
                    raise ValueError(f"Missing required field: {field}")
            return True
