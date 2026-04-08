"""PostgreSQL database schema extractor."""

from __future__ import annotations

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

    def extract_schema(
        self,
        schema_name: str = "public",
        include_tables: list[str] | None = None,
    ) -> list[MetadataArtifact]:
        """Extract complete PostgreSQL schema metadata.

        Args:
            schema_name: PostgreSQL schema to extract (default: "public").
            include_tables: If provided, only extract these tables. Extracts
                all tables when None or empty.

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

        # Get all tables in schema, then filter if requested
        table_names = self._get_inspector().get_table_names(schema=schema_name)
        if include_tables:
            include_set = set(include_tables)
            table_names = [t for t in table_names if t in include_set]

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
        artifacts.extend(self._extract_relationships(schema_name, include_tables=include_tables))

        # 6. Extract views
        artifacts.extend(self._extract_views(schema_name))

        # 7. Extract materialized views
        artifacts.extend(self._extract_materialized_views(schema_name))

        # 8. Extract functions and procedures
        artifacts.extend(self._extract_functions(schema_name))

        # 9. Extract sequences
        artifacts.extend(self._extract_sequences(schema_name))

        # 10. Extract custom types
        artifacts.extend(self._extract_types(schema_name))

        # 11. Extract triggers
        artifacts.extend(self._extract_triggers(schema_name))

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
        unique_columns: set[str] = set()
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
            constraints: list[str] = []
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
                        "foreign_key": fk_map.get(col_name, ""),  # type: ignore[reportUnknownMemberType]  # sqlalchemy
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

        # SQLAlchemy's get_indexes() excludes primary key indexes; include them explicitly.
        pk_constraint = self._get_inspector().get_pk_constraint(table_name, schema=schema)
        pk_index_name: str | None = pk_constraint.get("name")
        pk_columns: list[str] = pk_constraint.get("constrained_columns", [])
        if pk_index_name and pk_columns:
            artifact_id = self._generate_id(f"{schema}.{table_name}.{pk_index_name}", "index")
            artifacts.append(
                MetadataArtifact(
                    id=artifact_id,
                    name=pk_index_name,
                    type="index",
                    source_type="database",
                    language="postgresql",
                    module=f"{schema}.{table_name}",
                    description=f"Primary key index {pk_index_name} on table {table_name}",
                    tags=["database", "index", "primary_key", "postgresql"],
                    metadata={
                        "table_name": table_name,
                        "schema": schema,
                        "unique": True,
                        "columns": pk_columns,
                        "primary_key": True,
                    },
                )
            )

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
            check_constraints = self._get_inspector().get_check_constraints(
                table_name, schema=schema
            )

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

    def _extract_relationships(
        self,
        schema: str,
        include_tables: list[str] | None = None,
    ) -> list[MetadataArtifact]:
        """Extract foreign key relationships for all tables in schema.

        Args:
            schema: Schema name.
            include_tables: If provided, only include relationships where both
                the source and referenced table are in this list.

        Returns:
            List of MetadataArtifact objects for foreign key relationships.
        """
        artifacts: list[MetadataArtifact] = []

        table_names = self._get_inspector().get_table_names(schema=schema)
        if include_tables:
            include_set = set(include_tables)
            table_names = [t for t in table_names if t in include_set]

        for table_name in table_names:
            foreign_keys = self._get_inspector().get_foreign_keys(table_name, schema=schema)

            for fk in foreign_keys:
                fk_name = fk["name"]
                if fk_name is None:
                    continue  # Skip foreign keys without names
                referred_table = fk["referred_table"]
                if include_tables and referred_table not in include_tables:
                    continue  # Skip FK to tables outside the include list
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

    def _extract_views(self, schema: str) -> list[MetadataArtifact]:
        """Extract view metadata for the schema.

        Args:
            schema: Schema name.

        Returns:
            List of MetadataArtifact objects for each view.
        """
        artifacts: list[MetadataArtifact] = []

        view_names = self._get_inspector().get_view_names(schema=schema)

        for view_name in view_names:
            try:
                definition = (
                    self._get_inspector().get_view_definition(view_name, schema=schema) or ""
                )
            except Exception:
                definition = ""

            comment = self._get_table_comment(view_name, schema)
            artifact_id = self._generate_id(f"{schema}.{view_name}", "view")

            artifacts.append(
                MetadataArtifact(
                    id=artifact_id,
                    name=view_name,
                    type="view",
                    source_type="database",
                    language="postgresql",
                    module=schema,
                    description=comment or f"View {view_name} in schema {schema}",
                    tags=["database", "view", "postgresql"],
                    metadata={
                        "schema": schema,
                        "definition": definition,
                    },
                )
            )

        return artifacts

    def _extract_materialized_views(self, schema: str) -> list[MetadataArtifact]:
        """Extract materialized view metadata for the schema.

        Args:
            schema: Schema name.

        Returns:
            List of MetadataArtifact objects for each materialized view.
        """
        if not self.engine:
            return []

        artifacts: list[MetadataArtifact] = []

        try:
            query = text("""
                SELECT
                    mv.matviewname,
                    mv.definition,
                    obj_description(c.oid, 'pg_class') AS comment
                FROM pg_matviews mv
                JOIN pg_class c ON c.relname = mv.matviewname
                JOIN pg_namespace n ON n.oid = c.relnamespace AND n.nspname = :schema
                WHERE mv.schemaname = :schema
                ORDER BY mv.matviewname
            """)

            with self.engine.connect() as conn:
                result = conn.execute(query, {"schema": schema})
                for row in result:
                    name: str = row[0]
                    definition: str = row[1] or ""
                    comment: str = row[2] or ""

                    artifact_id = self._generate_id(f"{schema}.{name}", "materialized_view")

                    artifacts.append(
                        MetadataArtifact(
                            id=artifact_id,
                            name=name,
                            type="materialized_view",
                            source_type="database",
                            language="postgresql",
                            module=schema,
                            description=comment or f"Materialized view {name} in schema {schema}",
                            tags=["database", "materialized_view", "postgresql"],
                            metadata={
                                "schema": schema,
                                "definition": definition,
                            },
                        )
                    )
        except Exception:
            pass

        return artifacts

    def _extract_functions(self, schema: str) -> list[MetadataArtifact]:
        """Extract function and procedure metadata for the schema.

        Args:
            schema: Schema name.

        Returns:
            List of MetadataArtifact objects for each function/procedure.
        """
        if not self.engine:
            return []

        artifacts: list[MetadataArtifact] = []

        try:
            query = text("""
                SELECT
                    p.proname AS name,
                    pg_get_function_arguments(p.oid) AS arguments,
                    pg_get_function_result(p.oid) AS return_type,
                    pg_get_functiondef(p.oid) AS definition,
                    obj_description(p.oid, 'pg_proc') AS comment,
                    p.prokind AS kind
                FROM pg_proc p
                JOIN pg_namespace n ON n.oid = p.pronamespace
                WHERE n.nspname = :schema
                  AND p.prokind IN ('f', 'p')
                ORDER BY p.proname
            """)

            with self.engine.connect() as conn:
                result = conn.execute(query, {"schema": schema})
                for row in result:
                    name: str = row[0]
                    arguments: str = row[1] or ""
                    return_type: str = row[2] or ""
                    definition: str = row[3] or ""
                    comment: str = row[4] or ""
                    kind: str = row[5]

                    func_type = "procedure" if kind == "p" else "function"
                    artifact_id = self._generate_id(f"{schema}.{name}({arguments})", func_type)

                    artifacts.append(
                        MetadataArtifact(
                            id=artifact_id,
                            name=name,
                            type=func_type,
                            source_type="database",
                            language="postgresql",
                            module=schema,
                            description=comment
                            or f"{func_type.capitalize()} {name} in schema {schema}",
                            tags=["database", func_type, "postgresql"],
                            metadata={
                                "schema": schema,
                                "arguments": arguments,
                                "return_type": return_type,
                                "definition": definition,
                            },
                        )
                    )
        except Exception:
            pass

        return artifacts

    def _extract_sequences(self, schema: str) -> list[MetadataArtifact]:
        """Extract sequence metadata for the schema.

        Args:
            schema: Schema name.

        Returns:
            List of MetadataArtifact objects for each sequence.
        """
        if not self.engine:
            return []

        artifacts: list[MetadataArtifact] = []

        try:
            query = text("""
                SELECT
                    s.sequence_name,
                    s.data_type,
                    s.start_value,
                    s.minimum_value,
                    s.maximum_value,
                    s.increment,
                    s.cycle_option,
                    obj_description(c.oid, 'pg_class') AS comment
                FROM information_schema.sequences s
                JOIN pg_class c ON c.relname = s.sequence_name
                JOIN pg_namespace n ON n.oid = c.relnamespace AND n.nspname = :schema
                WHERE s.sequence_schema = :schema
                ORDER BY s.sequence_name
            """)

            with self.engine.connect() as conn:
                result = conn.execute(query, {"schema": schema})
                for row in result:
                    name: str = row[0]
                    data_type: str = row[1] or "bigint"
                    start_value: str = str(row[2])
                    min_value: str = str(row[3])
                    max_value: str = str(row[4])
                    increment: str = str(row[5])
                    cycle: bool = row[6] == "YES"
                    comment: str = row[7] or ""

                    artifact_id = self._generate_id(f"{schema}.{name}", "sequence")

                    artifacts.append(
                        MetadataArtifact(
                            id=artifact_id,
                            name=name,
                            type="sequence",
                            source_type="database",
                            language="postgresql",
                            module=schema,
                            description=comment or f"Sequence {name} in schema {schema}",
                            tags=["database", "sequence", "postgresql"],
                            metadata={
                                "schema": schema,
                                "data_type": data_type,
                                "start_value": start_value,
                                "minimum_value": min_value,
                                "maximum_value": max_value,
                                "increment": increment,
                                "cycle": cycle,
                            },
                        )
                    )
        except Exception:
            pass

        return artifacts

    def _extract_types(self, schema: str) -> list[MetadataArtifact]:
        """Extract custom data type metadata for the schema.

        Only extracts user-defined types (enums, composite types, domains,
        range types). Excludes built-in types and auto-generated array types.

        Args:
            schema: Schema name.

        Returns:
            List of MetadataArtifact objects for each custom type.
        """
        if not self.engine:
            return []

        artifacts: list[MetadataArtifact] = []

        try:
            query = text("""
                SELECT
                    t.typname AS name,
                    t.typtype AS kind,
                    obj_description(t.oid, 'pg_type') AS comment,
                    CASE
                        WHEN t.typtype = 'e' THEN (
                            SELECT string_agg(e.enumlabel, ', ' ORDER BY e.enumsortorder)
                            FROM pg_enum e WHERE e.enumtypid = t.oid
                        )
                        WHEN t.typtype = 'c' THEN (
                            SELECT string_agg(
                                a.attname || ' ' || pg_catalog.format_type(a.atttypid, a.atttypmod),
                                ', '
                                ORDER BY a.attnum
                            )
                            FROM pg_attribute a
                            WHERE a.attrelid = t.typrelid
                              AND a.attnum > 0
                              AND NOT a.attisdropped
                        )
                        WHEN t.typtype = 'd' THEN pg_catalog.format_type(t.typbasetype, t.typtypmod)
                        ELSE NULL
                    END AS type_details
                FROM pg_type t
                JOIN pg_namespace n ON n.oid = t.typnamespace
                WHERE n.nspname = :schema
                  AND t.typtype IN ('e', 'c', 'd', 'r')
                  AND left(t.typname, 1) <> '_'
                  AND (
                      t.typtype <> 'c'
                      OR (SELECT relkind FROM pg_class c WHERE c.oid = t.typrelid) = 'c'
                  )
                ORDER BY t.typname
            """)

            kind_labels: dict[str, str] = {
                "e": "enum",
                "c": "composite",
                "d": "domain",
                "r": "range",
            }

            with self.engine.connect() as conn:
                result = conn.execute(query, {"schema": schema})
                for row in result:
                    name: str = row[0]
                    kind: str = row[1]
                    comment: str = row[2] or ""
                    type_details: str = row[3] or ""

                    kind_label = kind_labels.get(kind, kind)
                    artifact_id = self._generate_id(f"{schema}.{name}", "type")

                    artifacts.append(
                        MetadataArtifact(
                            id=artifact_id,
                            name=name,
                            type="type",
                            source_type="database",
                            language="postgresql",
                            module=schema,
                            description=comment
                            or f"{kind_label.capitalize()} type {name} in schema {schema}",
                            tags=["database", "type", kind_label, "postgresql"],
                            metadata={
                                "schema": schema,
                                "type_kind": kind_label,
                                "details": type_details,
                            },
                        )
                    )
        except Exception:
            pass

        return artifacts

    def _extract_triggers(self, schema: str) -> list[MetadataArtifact]:
        """Extract trigger metadata for the schema.

        Args:
            schema: Schema name.

        Returns:
            List of MetadataArtifact objects for each trigger.
        """
        if not self.engine:
            return []

        artifacts: list[MetadataArtifact] = []

        try:
            query = text("""
                SELECT
                    trigger_name,
                    event_object_table,
                    action_timing,
                    string_agg(event_manipulation, ' OR ' ORDER BY event_manipulation) AS events,
                    action_orientation,
                    action_statement
                FROM information_schema.triggers
                WHERE trigger_schema = :schema
                GROUP BY
                    trigger_name,
                    event_object_table,
                    action_timing,
                    action_orientation,
                    action_statement
                ORDER BY event_object_table, trigger_name
            """)

            with self.engine.connect() as conn:
                result = conn.execute(query, {"schema": schema})
                for row in result:
                    name: str = row[0]
                    table_name: str = row[1]
                    timing: str = row[2]
                    events: str = row[3] or ""
                    orientation: str = row[4]
                    action: str = row[5] or ""

                    artifact_id = self._generate_id(f"{schema}.{table_name}.{name}", "trigger")
                    description = (
                        f"{timing} {events} trigger {name} on {table_name} in schema {schema}"
                    )

                    artifacts.append(
                        MetadataArtifact(
                            id=artifact_id,
                            name=name,
                            type="trigger",
                            source_type="database",
                            language="postgresql",
                            module=f"{schema}.{table_name}",
                            description=description,
                            tags=["database", "trigger", "postgresql"],
                            metadata={
                                "schema": schema,
                                "table_name": table_name,
                                "timing": timing,
                                "events": events,
                                "orientation": orientation,
                                "action": action,
                            },
                        )
                    )
        except Exception:
            pass

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
