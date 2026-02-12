"""PostgreSQL database schema extractor."""

import hashlib
from typing import Any

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine, Inspector

from .base import MetadataArtifact
from .database_extractor import DatabaseExtractor


class PostgresExtractor(DatabaseExtractor):
    """Extract schema metadata from PostgreSQL databases.

    Uses SQLAlchemy Inspector for comprehensive schema extraction,
    supplemented with direct queries to pg_catalog for comments.
    """

    def __init__(self):
        """Initialize PostgreSQL extractor."""
        super().__init__()
        self.engine: Engine | None = None
        self.inspector: Inspector | None = None

    def _get_inspector(self) -> Inspector:
        """Get inspector, raising error if not connected."""
        if self.inspector is None:
            raise RuntimeError("Not connected to database. Call connect() first.")
        return self.inspector

    def connect(self, connection_config: dict[str, Any]) -> None:
        """Establish PostgreSQL connection.

        Args:
            connection_config: PostgreSQL connection parameters.
                - host: Database host (default: localhost)
                - port: Database port (default: 5432)
                - database: Database name
                - user: Username
                - password: Password

        Raises:
            ConnectionError: If connection fails.
        """
        host = connection_config.get("host", "localhost")
        port = connection_config.get("port", 5432)
        database = connection_config["database"]
        user = connection_config["user"]
        password = connection_config["password"]

        # Build connection URL
        url = f"postgresql://{user}:{password}@{host}:{port}/{database}"

        try:
            self.engine = create_engine(url)
            self.inspector = inspect(self.engine)
            self._connected = True
            self.connection_config = connection_config
        except Exception as e:
            raise ConnectionError(f"Failed to connect to PostgreSQL: {e}") from e

    def disconnect(self) -> None:
        """Close PostgreSQL connection."""
        if self.engine:
            self.engine.dispose()
            self.engine = None
            self.inspector = None
            self._connected = False

    def extract_schema(self, schema_name: str = "public") -> list[MetadataArtifact]:
        """Extract complete PostgreSQL schema metadata.

        Args:
            schema_name: PostgreSQL schema to extract (default: "public").

        Returns:
            List of metadata artifacts for tables, columns, indexes,
            constraints, and relationships.

        Raises:
            RuntimeError: If not connected to database.
        """
        if not self._connected or not self.inspector:
            raise RuntimeError("Not connected to database. Call connect() first.")

        # Type narrowing for inspector
        assert self.inspector is not None

        artifacts: list[MetadataArtifact] = []

        # Get all tables in schema
        table_names = self._get_inspector().get_table_names(schema=schema_name)

        for table_name in table_names:
            # 1. Extract table artifact
            artifacts.append(self._extract_table(table_name, schema_name))

            # 2. Extract columns for this table
            artifacts.extend(self._extract_columns(table_name, schema_name))

            # 3. Extract indexes
            artifacts.extend(self._extract_indexes(table_name, schema_name))

            # 4. Extract constraints (unique, check)
            artifacts.extend(self._extract_constraints(table_name, schema_name))

        # 5. Extract relationships (foreign keys) - done after all tables
        artifacts.extend(self._extract_relationships(schema_name))

        return artifacts

    def _extract_table(self, table_name: str, schema: str) -> MetadataArtifact:
        """Extract table metadata.

        Args:
            table_name: Table name.
            schema: Schema name.

        Returns:
            MetadataArtifact for the table.
        """
        # Get table comment from PostgreSQL catalog
        comment = self._get_table_comment(table_name, schema)

        # Generate unique ID
        artifact_id = self._generate_id(f"{schema}.{table_name}", "table")

        return MetadataArtifact(
            id=artifact_id,
            name=table_name,
            type="table",
            source_type="database",
            language="postgresql",
            module=schema,
            description=comment or f"Table {table_name} in schema {schema}",
            tags=["database", "table", "postgresql"],
            metadata={
                "schema": schema,
                "table_type": "BASE TABLE",  # Could check for views
            },
        )

    def _extract_columns(self, table_name: str, schema: str) -> list[MetadataArtifact]:
        """Extract column metadata for a table.

        Args:
            table_name: Table name.
            schema: Schema name.

        Returns:
            List of MetadataArtifact objects for each column.
        """
        artifacts: list[MetadataArtifact] = []

        # Get column details from inspector
        columns = self._get_inspector().get_columns(table_name, schema=schema)

        # Get primary key constraint
        pk_constraint = self._get_inspector().get_pk_constraint(table_name, schema=schema)
        pk_columns = pk_constraint.get("constrained_columns", [])

        # Get foreign keys to mark FK columns
        foreign_keys = self._get_inspector().get_foreign_keys(table_name, schema=schema)
        fk_map = {}  # column_name -> referenced_table.column
        for fk in foreign_keys:
            for i, col in enumerate(fk["constrained_columns"]):
                ref_table = fk["referred_table"]
                ref_col = fk["referred_columns"][i]
                fk_map[col] = f"{ref_table}.{ref_col}"

        # Get unique constraints
        unique_constraints = self._get_inspector().get_unique_constraints(table_name, schema=schema)
        unique_columns = set()
        for uc in unique_constraints:
            unique_columns.update(uc.get("column_names", []))

        for col in columns:
            col_name = col["name"]
            col_type = str(col["type"])
            nullable = col["nullable"]
            default = col.get("default", "")

            # Get column comment
            comment = col.get("comment", "")

            # Build constraints list
            constraints = []
            if not nullable:
                constraints.append("NOT NULL")
            if col_name in pk_columns:
                constraints.append("PRIMARY KEY")
            if col_name in unique_columns and col_name not in pk_columns:
                constraints.append("UNIQUE")

            # Generate unique ID
            artifact_id = self._generate_id(f"{schema}.{table_name}.{col_name}", "column")

            artifacts.append(
                MetadataArtifact(
                    id=artifact_id,
                    name=col_name,
                    type="column",
                    source_type="database",
                    language="postgresql",
                    module=f"{schema}.{table_name}",
                    description=comment or f"Column {col_name} in table {table_name}",
                    constraints=constraints,
                    tags=["database", "column", "postgresql"],
                    metadata={
                        "table_name": table_name,
                        "schema": schema,
                        "column_type": col_type,
                        "nullable": nullable,
                        "default_value": str(default) if default else "",
                        "primary_key": col_name in pk_columns,
                        "foreign_key": fk_map.get(col_name, ""),
                        "unique": col_name in unique_columns,
                    },
                )
            )

        return artifacts

    def _extract_indexes(self, table_name: str, schema: str) -> list[MetadataArtifact]:
        """Extract index metadata for a table.

        Args:
            table_name: Table name.
            schema: Schema name.

        Returns:
            List of MetadataArtifact objects for each index.
        """
        artifacts: list[MetadataArtifact] = []

        indexes = self._get_inspector().get_indexes(table_name, schema=schema)

        for idx in indexes:
            idx_name = idx["name"]
            if idx_name is None:
                continue  # Skip indexes without names
            unique = idx.get("unique", False)
            columns = idx.get("column_names", [])

            # Generate unique ID
            artifact_id = self._generate_id(f"{schema}.{table_name}.{idx_name}", "index")

            artifacts.append(
                MetadataArtifact(
                    id=artifact_id,
                    name=idx_name,
                    type="index",
                    source_type="database",
                    language="postgresql",
                    module=f"{schema}.{table_name}",
                    description=f"Index {idx_name} on table {table_name}",
                    tags=["database", "index", "postgresql"],
                    metadata={
                        "table_name": table_name,
                        "schema": schema,
                        "unique": unique,
                        "columns": columns,
                    },
                )
            )

        return artifacts

    def _extract_constraints(self, table_name: str, schema: str) -> list[MetadataArtifact]:
        """Extract constraint metadata for a table.

        Args:
            table_name: Table name.
            schema: Schema name.

        Returns:
            List of MetadataArtifact objects for constraints.
        """
        artifacts: list[MetadataArtifact] = []

        # Get check constraints
        try:
            check_constraints = self._get_inspector().get_check_constraints(table_name, schema=schema)

            for chk in check_constraints:
                chk_name = chk["name"]
                if chk_name is None:
                    continue  # Skip constraints without names
                sqltext = chk.get("sqltext", "")

                # Generate unique ID
                artifact_id = self._generate_id(f"{schema}.{table_name}.{chk_name}", "constraint")

                artifacts.append(
                    MetadataArtifact(
                        id=artifact_id,
                        name=chk_name,
                        type="constraint",
                        source_type="database",
                        language="postgresql",
                        module=f"{schema}.{table_name}",
                        description=f"Check constraint: {sqltext}",
                        constraints=[sqltext],
                        tags=["database", "constraint", "check", "postgresql"],
                        metadata={
                            "table_name": table_name,
                            "schema": schema,
                            "constraint_type": "CHECK",
                            "sqltext": sqltext,
                        },
                    )
                )
        except NotImplementedError:
            # Some database backends don't support check constraints
            pass

        return artifacts

    def _extract_relationships(self, schema: str) -> list[MetadataArtifact]:
        """Extract foreign key relationships for all tables in schema.

        Args:
            schema: Schema name.

        Returns:
            List of MetadataArtifact objects for foreign key relationships.
        """
        artifacts: list[MetadataArtifact] = []

        table_names = self._get_inspector().get_table_names(schema=schema)

        for table_name in table_names:
            foreign_keys = self._get_inspector().get_foreign_keys(table_name, schema=schema)

            for fk in foreign_keys:
                fk_name = fk["name"]
                if fk_name is None:
                    continue  # Skip foreign keys without names
                referred_table = fk["referred_table"]
                constrained_cols = fk["constrained_columns"]
                referred_cols = fk["referred_columns"]

                # Build relation strings
                relations: list[str] = []
                for i, col in enumerate(constrained_cols):
                    ref_col = referred_cols[i] if i < len(referred_cols) else "?"
                    relations.append(f"{table_name}.{col} -> {referred_table}.{ref_col}")

                # Generate unique ID
                artifact_id = self._generate_id(f"{schema}.{fk_name}", "relationship")

                artifacts.append(
                    MetadataArtifact(
                        id=artifact_id,
                        name=fk_name,
                        type="relationship",
                        source_type="database",
                        language="postgresql",
                        module=schema,
                        description=f"Foreign key from {table_name} to {referred_table}",
                        relations=relations,
                        tags=["database", "foreign_key", "relationship", "postgresql"],
                        metadata={
                            "from_table": table_name,
                            "to_table": referred_table,
                            "from_columns": constrained_cols,
                            "to_columns": referred_cols,
                            "cardinality": "N:1",  # Standard FK cardinality
                            "schema": schema,
                        },
                    )
                )

        return artifacts

    def _get_table_comment(self, table_name: str, schema: str) -> str:
        """Get table comment from PostgreSQL catalog.

        Args:
            table_name: Table name.
            schema: Schema name.

        Returns:
            Table comment or empty string.
        """
        if not self.engine:
            return ""

        try:
            query = text("""
                SELECT obj_description(
                    (quote_ident(:schema) || '.' || quote_ident(:table))::regclass,
                    'pg_class'
                ) AS comment
            """)

            with self.engine.connect() as conn:
                result = conn.execute(query, {"schema": schema, "table": table_name})
                row = result.fetchone()
                if row and row[0]:
                    return row[0]
        except Exception:
            # Ignore errors and return empty string
            pass

        return ""

    def _generate_id(self, qualified_name: str, artifact_type: str) -> str:
        """Generate unique artifact ID.

        Args:
            qualified_name: Fully qualified name (e.g., "public.sale_order.id").
            artifact_type: Artifact type (e.g., "table", "column").

        Returns:
            Unique identifier for the artifact.
        """
        # Use hash of qualified name + type + language
        content = f"postgresql:{artifact_type}:{qualified_name}"
        hash_digest = hashlib.sha256(content.encode()).hexdigest()[:16]
        return f"pg_{artifact_type}_{hash_digest}"

    def language(self) -> str:
        """Return the database type identifier.

        Returns:
            "postgresql"
        """
        return "postgresql"
