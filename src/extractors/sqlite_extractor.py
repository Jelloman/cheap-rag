"""SQLite database schema extractor."""

import hashlib
from typing import Any

from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import Engine, Inspector

from .base import MetadataArtifact
from .database_extractor import DatabaseExtractor


class SqliteExtractor(DatabaseExtractor):
    """Extract schema metadata from SQLite databases.

    SQLite is simpler than PostgreSQL:
    - No schemas (all tables in main database)
    - Simpler type system
    - Fewer constraint types

    Good for testing and lightweight examples.
    """

    def __init__(self):
        """Initialize SQLite extractor."""
        super().__init__()
        self.engine: Engine | None = None
        self.inspector: Inspector | None = None
        self.database_path: str = ""

    def _get_inspector(self) -> Inspector:
        """Get inspector, raising error if not connected."""
        if self.inspector is None:
            raise RuntimeError("Not connected to database. Call connect() first.")
        return self.inspector

    def connect(self, connection_config: dict[str, Any]) -> None:
        """Establish SQLite connection.

        Args:
            connection_config: SQLite connection parameters.
                - path: Path to SQLite database file

        Raises:
            ConnectionError: If connection fails.
        """
        self.database_path = connection_config["path"]

        # Build connection URL
        url = f"sqlite:///{self.database_path}"

        try:
            self.engine = create_engine(url)
            self.inspector = inspect(self.engine)
            self._connected = True
            self.connection_config = connection_config
        except Exception as e:
            raise ConnectionError(f"Failed to connect to SQLite: {e}") from e

    def disconnect(self) -> None:
        """Close SQLite connection."""
        if self.engine:
            self.engine.dispose()
            self.engine = None
            self.inspector = None
            self._connected = False

    def extract_schema(self, schema_name: str = "main") -> list[MetadataArtifact]:  # noqa: ARG002
        """Extract complete SQLite schema metadata.

        Args:
            schema_name: Ignored for SQLite (no schema support, always "main").

        Returns:
            List of metadata artifacts for tables, columns, indexes,
            and relationships.

        Raises:
            RuntimeError: If not connected to database.
        """
        if not self._connected or not self.inspector:
            raise RuntimeError("Not connected to database. Call connect() first.")

        # Type narrowing for inspector
        assert self.inspector is not None

        artifacts: list[MetadataArtifact] = []

        # Get all tables (SQLite has no schemas, use None)
        table_names = self._get_inspector().get_table_names()

        for table_name in table_names:
            # Skip SQLite internal tables
            if table_name.startswith("sqlite_"):
                continue

            # 1. Extract table artifact
            artifacts.append(self._extract_table(table_name))

            # 2. Extract columns for this table
            artifacts.extend(self._extract_columns(table_name))

            # 3. Extract indexes
            artifacts.extend(self._extract_indexes(table_name))

        # 4. Extract relationships (foreign keys)
        artifacts.extend(self._extract_relationships())

        return artifacts

    def _extract_table(self, table_name: str) -> MetadataArtifact:
        """Extract table metadata.

        Args:
            table_name: Table name.

        Returns:
            MetadataArtifact for the table.
        """
        # Generate unique ID
        artifact_id = self._generate_id(table_name, "table")

        return MetadataArtifact(
            id=artifact_id,
            name=table_name,
            type="table",
            source_type="database",
            language="sqlite",
            module="main",  # SQLite default schema
            description=f"Table {table_name}",
            tags=["database", "table", "sqlite"],
            metadata={
                "schema": "main",
                "table_type": "BASE TABLE",
                "database_path": self.database_path,
            },
        )

    def _extract_columns(self, table_name: str) -> list[MetadataArtifact]:
        """Extract column metadata for a table.

        Args:
            table_name: Table name.

        Returns:
            List of MetadataArtifact objects for each column.
        """
        artifacts: list[MetadataArtifact] = []

        # Get column details from inspector
        columns = self._get_inspector().get_columns(table_name)

        # Get primary key constraint
        pk_constraint = self._get_inspector().get_pk_constraint(table_name)
        pk_columns = pk_constraint.get("constrained_columns", [])

        # Get foreign keys to mark FK columns
        foreign_keys = self._get_inspector().get_foreign_keys(table_name)
        fk_map = {}  # column_name -> referenced_table.column
        for fk in foreign_keys:
            for i, col in enumerate(fk["constrained_columns"]):
                ref_table = fk["referred_table"]
                ref_col = fk["referred_columns"][i]
                fk_map[col] = f"{ref_table}.{ref_col}"

        for col in columns:
            col_name = col["name"]
            col_type = str(col["type"])
            nullable = col["nullable"]
            default = col.get("default", "")

            # Build constraints list
            constraints: list[str] = []
            if not nullable:
                constraints.append("NOT NULL")
            if col_name in pk_columns:
                constraints.append("PRIMARY KEY")

            # Generate unique ID
            artifact_id = self._generate_id(f"{table_name}.{col_name}", "column")

            artifacts.append(
                MetadataArtifact(
                    id=artifact_id,
                    name=col_name,
                    type="column",
                    source_type="database",
                    language="sqlite",
                    module=f"main.{table_name}",
                    description=f"Column {col_name} in table {table_name}",
                    constraints=constraints,
                    tags=["database", "column", "sqlite"],
                    metadata={
                        "table_name": table_name,
                        "schema": "main",
                        "column_type": col_type,
                        "nullable": nullable,
                        "default_value": str(default) if default else "",
                        "primary_key": col_name in pk_columns,
                        "foreign_key": fk_map.get(col_name, ""),
                        "database_path": self.database_path,
                    },
                )
            )

        return artifacts

    def _extract_indexes(self, table_name: str) -> list[MetadataArtifact]:
        """Extract index metadata for a table.

        Args:
            table_name: Table name.

        Returns:
            List of MetadataArtifact objects for each index.
        """
        artifacts: list[MetadataArtifact] = []

        indexes = self._get_inspector().get_indexes(table_name)

        for idx in indexes:
            idx_name = idx["name"]
            if idx_name is None:
                continue  # Skip indexes without names
            unique = idx.get("unique", False)
            columns = idx.get("column_names", [])

            # Generate unique ID
            artifact_id = self._generate_id(f"{table_name}.{idx_name}", "index")

            artifacts.append(
                MetadataArtifact(
                    id=artifact_id,
                    name=idx_name,
                    type="index",
                    source_type="database",
                    language="sqlite",
                    module=f"main.{table_name}",
                    description=f"Index {idx_name} on table {table_name}",
                    tags=["database", "index", "sqlite"],
                    metadata={
                        "table_name": table_name,
                        "schema": "main",
                        "unique": unique,
                        "columns": columns,
                        "database_path": self.database_path,
                    },
                )
            )

        return artifacts

    def _extract_relationships(self) -> list[MetadataArtifact]:
        """Extract foreign key relationships for all tables.

        Returns:
            List of MetadataArtifact objects for foreign key relationships.
        """
        artifacts: list[MetadataArtifact] = []

        table_names = self._get_inspector().get_table_names()

        for table_name in table_names:
            # Skip SQLite internal tables
            if table_name.startswith("sqlite_"):
                continue

            foreign_keys = self._get_inspector().get_foreign_keys(table_name)

            for fk in foreign_keys:
                fk_name: str = fk.get("name") or f"{table_name}_fk"
                referred_table = fk["referred_table"]
                constrained_cols = fk["constrained_columns"]
                referred_cols = fk["referred_columns"]

                # Build relation strings
                relations: list[str] = []
                for i, col in enumerate(constrained_cols):
                    ref_col = referred_cols[i] if i < len(referred_cols) else "?"
                    relations.append(f"{table_name}.{col} -> {referred_table}.{ref_col}")

                # Generate unique ID
                artifact_id = self._generate_id(fk_name, "relationship")

                artifacts.append(
                    MetadataArtifact(
                        id=artifact_id,
                        name=fk_name,
                        type="relationship",
                        source_type="database",
                        language="sqlite",
                        module="main",
                        description=f"Foreign key from {table_name} to {referred_table}",
                        relations=relations,
                        tags=["database", "foreign_key", "relationship", "sqlite"],
                        metadata={
                            "from_table": table_name,
                            "to_table": referred_table,
                            "from_columns": constrained_cols,
                            "to_columns": referred_cols,
                            "cardinality": "N:1",
                            "schema": "main",
                            "database_path": self.database_path,
                        },
                    )
                )

        return artifacts

    def _generate_id(self, qualified_name: str, artifact_type: str) -> str:
        """Generate unique artifact ID.

        Args:
            qualified_name: Qualified name (e.g., "sale_order.id").
            artifact_type: Artifact type (e.g., "table", "column").

        Returns:
            Unique identifier for the artifact.
        """
        # Use hash of qualified name + type + language + database path
        content = f"sqlite:{artifact_type}:{self.database_path}:{qualified_name}"
        hash_digest = hashlib.sha256(content.encode()).hexdigest()[:16]
        return f"sqlite_{artifact_type}_{hash_digest}"

    def language(self) -> str:
        """Return the database type identifier.

        Returns:
            "sqlite"
        """
        return "sqlite"
