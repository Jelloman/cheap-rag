"""Tests for base metadata artifact functionality."""

import pytest

from src.extractors.base import MetadataArtifact


class TestMetadataArtifact:
    """Test MetadataArtifact class with database and code artifacts."""

    def test_database_table_artifact(self):
        """Test creating a database table artifact."""
        artifact = MetadataArtifact(
            id="pg_table_abc123",
            name="sale_order",
            type="table",
            source_type="database",
            language="postgresql",
            module="public",
            description="Sales order table",
            tags=["database", "table", "sales"],
            metadata={
                "schema": "public",
                "table_type": "BASE TABLE",
            },
        )

        assert artifact.type == "table"
        assert artifact.source_type == "database"
        assert artifact.language == "postgresql"
        assert artifact.metadata["schema"] == "public"

    def test_database_column_artifact(self):
        """Test creating a database column artifact."""
        artifact = MetadataArtifact(
            id="pg_column_xyz789",
            name="customer_id",
            type="column",
            source_type="database",
            language="postgresql",
            module="public.sale_order",
            description="Customer reference",
            constraints=["NOT NULL"],
            tags=["database", "column", "foreign_key"],
            metadata={
                "table_name": "sale_order",
                "schema": "public",
                "column_type": "BIGINT",
                "nullable": False,
                "primary_key": False,
                "foreign_key": "res_partner.id",
            },
        )

        assert artifact.type == "column"
        assert artifact.metadata["column_type"] == "BIGINT"
        assert artifact.metadata["foreign_key"] == "res_partner.id"
        assert not artifact.metadata["nullable"]

    def test_database_relationship_artifact(self):
        """Test creating a database relationship artifact."""
        artifact = MetadataArtifact(
            id="pg_relationship_rel123",
            name="sale_order_customer_fkey",
            type="relationship",
            source_type="database",
            language="postgresql",
            module="public",
            description="FK from sale_order to res_partner",
            relations=["sale_order.customer_id -> res_partner.id"],
            tags=["database", "foreign_key", "relationship"],
            metadata={
                "from_table": "sale_order",
                "to_table": "res_partner",
                "from_columns": ["customer_id"],
                "to_columns": ["id"],
                "cardinality": "N:1",
            },
        )

        assert artifact.type == "relationship"
        assert artifact.metadata["cardinality"] == "N:1"
        assert len(artifact.relations) == 1

    def test_code_class_artifact(self):
        """Test creating a code class artifact."""
        artifact = MetadataArtifact(
            id="java_interface_catalog123",
            name="Catalog",
            type="interface",
            source_type="code",
            language="java",
            module="com.example.cheap.core",
            description="Root container for CHEAP data",
            tags=["java", "code", "interface", "core"],
            source_file="Catalog.java",
            source_line=15,
        )

        assert artifact.type == "interface"
        assert artifact.source_type == "code"
        assert artifact.source_file == "Catalog.java"
        assert artifact.source_line == 15

    def test_to_dict_includes_source_type(self):
        """Test that to_dict includes source_type field."""
        artifact = MetadataArtifact(
            id="test_id",
            name="test_table",
            type="table",
            source_type="database",
            language="postgresql",
            module="public",
            description="Test table",
        )

        artifact_dict = artifact.to_dict()

        assert "source_type" in artifact_dict
        assert artifact_dict["source_type"] == "database"

    def test_from_dict_with_source_type(self):
        """Test creating artifact from dict with source_type."""
        data = {
            "id": "test_id",
            "name": "test_table",
            "type": "table",
            "source_type": "database",
            "language": "postgresql",
            "module": "public",
            "description": "Test table",
            "constraints": [],
            "relations": [],
            "examples": [],
            "tags": [],
            "source_file": "",
            "source_line": 0,
            "metadata": {},
        }

        artifact = MetadataArtifact.from_dict(data)

        assert artifact.source_type == "database"
        assert artifact.type == "table"

    def test_database_table_embedding_text(self):
        """Test embedding text generation for database table."""
        artifact = MetadataArtifact(
            id="pg_table_abc",
            name="sale_order",
            type="table",
            source_type="database",
            language="postgresql",
            module="public",
            description="Sales orders from customers",
            tags=["sales", "ecommerce"],
        )

        embedding_text = artifact.to_embedding_text()

        assert "[TABLE]" in embedding_text
        assert "sale_order" in embedding_text
        assert "postgresql" in embedding_text
        assert "Schema: public" in embedding_text
        assert "Sales orders from customers" in embedding_text

    def test_database_column_embedding_text(self):
        """Test embedding text generation for database column."""
        artifact = MetadataArtifact(
            id="pg_column_xyz",
            name="customer_id",
            type="column",
            source_type="database",
            language="postgresql",
            module="public.sale_order",
            description="Customer reference",
            metadata={
                "table_name": "sale_order",
                "column_type": "BIGINT",
                "nullable": False,
                "primary_key": False,
                "foreign_key": "res_partner.id",
            },
        )

        embedding_text = artifact.to_embedding_text()

        assert "[COLUMN]" in embedding_text
        assert "customer_id" in embedding_text
        assert "sale_order" in embedding_text
        assert "Type: BIGINT" in embedding_text
        assert "Nullable: No" in embedding_text
        assert "Foreign Key to: res_partner.id" in embedding_text

    def test_database_relationship_embedding_text(self):
        """Test embedding text generation for database relationship."""
        artifact = MetadataArtifact(
            id="pg_rel_123",
            name="sale_order_customer_fkey",
            type="relationship",
            source_type="database",
            language="postgresql",
            module="public",
            description="Links orders to customers",
            relations=["sale_order.customer_id -> res_partner.id"],
            metadata={
                "from_table": "sale_order",
                "to_table": "res_partner",
                "cardinality": "N:1",
            },
        )

        embedding_text = artifact.to_embedding_text()

        assert "[RELATIONSHIP]" in embedding_text
        assert "sale_order" in embedding_text
        assert "res_partner" in embedding_text
        assert "Cardinality: N:1" in embedding_text

    def test_code_artifact_embedding_text(self):
        """Test embedding text generation for code artifact (backward compat)."""
        artifact = MetadataArtifact(
            id="java_class_123",
            name="Catalog",
            type="class",
            source_type="code",
            language="java",
            module="com.example.cheap",
            description="Root container",
            tags=["core", "interface"],
        )

        embedding_text = artifact.to_embedding_text()

        assert "[CLASS]" in embedding_text
        assert "Catalog" in embedding_text
        assert "java" in embedding_text
        assert "Module: com.example.cheap" in embedding_text
        assert "Root container" in embedding_text
