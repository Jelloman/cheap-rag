"""Base class for database schema extractors."""

from abc import abstractmethod
from pathlib import Path
from typing import Any

from .base import MetadataArtifact, MetadataExtractor


class DatabaseExtractor(MetadataExtractor):
    """Base class for database schema extractors.

    Database extractors connect to a database, extract schema metadata
    (tables, columns, indexes, constraints, relationships), and return
    MetadataArtifact objects suitable for embedding and indexing.
    """

    def __init__(self):
        """Initialize database extractor."""
        self.connection_config: dict[str, Any] = {}
        self._connected = False

    @abstractmethod
    def connect(self, connection_config: dict[str, Any]) -> None:
        """Establish database connection.

        Args:
            connection_config: Database connection parameters.
                - host: Database host
                - port: Database port
                - database: Database name
                - user: Username
                - password: Password
                - (other database-specific options)

        Raises:
            ConnectionError: If connection fails.
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Close database connection."""
        pass

    @abstractmethod
    def extract_schema(self, schema_name: str = "public") -> list[MetadataArtifact]:
        """Extract complete schema metadata.

        Returns artifacts for:
        - Tables (type="table")
        - Columns (type="column")
        - Indexes (type="index")
        - Constraints (type="constraint")
        - Foreign key relationships (type="relationship")

        Args:
            schema_name: Database schema to extract (default: "public").
                         For SQLite, this should be None.

        Returns:
            List of metadata artifacts representing the database schema.

        Raises:
            RuntimeError: If not connected to database.
        """
        pass

    def extract_metadata(self, source_path: Path) -> list[MetadataArtifact]:
        """Extract metadata from database connection config file.

        This is the MetadataExtractor interface method. It reads a
        connection config file (YAML or JSON) and calls connect() and
        extract_schema().

        Args:
            source_path: Path to connection config file.

        Returns:
            List of extracted metadata artifacts.
        """
        import json
        import yaml

        # Read connection config from file
        if source_path.suffix in [".yaml", ".yml"]:
            with open(source_path, "r") as f:
                config = yaml.safe_load(f)
        elif source_path.suffix == ".json":
            with open(source_path, "r") as f:
                config = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {source_path.suffix}")

        # Connect and extract
        self.connect(config["connection"])
        schema = config.get("schema", "public")
        artifacts = self.extract_schema(schema)
        self.disconnect()

        # Apply configured tags
        if "tags" in config:
            for artifact in artifacts:
                artifact.tags.extend(config["tags"])

        return artifacts

    @abstractmethod
    def language(self) -> str:
        """Return the database type identifier.

        Returns:
            Database identifier: "postgresql", "sqlite", "mariadb", etc.
        """
        pass
