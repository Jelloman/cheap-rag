"""Tests for PostgreSQL metadata extractor using testing.postgresql."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

import psycopg2  # type: ignore[import-untyped]
import pytest

try:
    import testing.postgresql  # type: ignore[import-untyped]
except ImportError:
    pytest.skip("testing.postgresql not installed", allow_module_level=True)

from src.extractors.postgres_extractor import PostgresExtractor


# ---------------------------------------------------------------------------
# Schema SQL
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
    -- Custom enum type
    CREATE TYPE mpaa_rating AS ENUM ('G', 'PG', 'PG-13', 'R', 'NC-17');

    -- Parent table: category
    CREATE TABLE category (
        category_id SERIAL PRIMARY KEY,
        name        VARCHAR(25) NOT NULL,
        last_update TIMESTAMP  NOT NULL DEFAULT NOW(),
        CONSTRAINT uq_category_name UNIQUE (name)
    );

    -- Film table: enum column + check constraint
    CREATE TABLE film (
        film_id     SERIAL PRIMARY KEY,
        title       VARCHAR(255)  NOT NULL,
        description TEXT,
        rating      mpaa_rating   DEFAULT 'G',
        length      SMALLINT,
        CONSTRAINT chk_film_length CHECK (length > 0)
    );

    -- Junction table: composite PK + two FKs
    CREATE TABLE film_category (
        film_id     INTEGER NOT NULL,
        category_id INTEGER NOT NULL,
        CONSTRAINT film_category_pkey PRIMARY KEY (film_id, category_id),
        CONSTRAINT fk_film_category_film
            FOREIGN KEY (film_id)     REFERENCES film(film_id),
        CONSTRAINT fk_film_category_category
            FOREIGN KEY (category_id) REFERENCES category(category_id)
    );

    -- Regular (non-unique) index on film.title
    CREATE INDEX idx_film_title ON film(title);

    -- View joining film and category
    CREATE VIEW film_in_category AS
        SELECT f.film_id, f.title, c.name AS category_name
        FROM   film f
        JOIN   film_category fc ON f.film_id     = fc.film_id
        JOIN   category      c  ON fc.category_id = c.category_id;

    -- Trigger function + trigger on category
    CREATE OR REPLACE FUNCTION trg_update_last_update()
    RETURNS TRIGGER LANGUAGE plpgsql AS $$
    BEGIN
        NEW.last_update = NOW();
        RETURN NEW;
    END;
    $$;

    CREATE TRIGGER trg_category_last_update
    BEFORE UPDATE ON category
    FOR EACH ROW EXECUTE FUNCTION trg_update_last_update();
"""


# ---------------------------------------------------------------------------
# Factory with cached initialization
# ---------------------------------------------------------------------------


def _initialize_db(postgresql: Any) -> None:  # type: ignore[explicit-any]
    """Populate the test schema; called once and baked into the cache."""
    conn: Any = psycopg2.connect(**postgresql.dsn())  # type: ignore[explicit-any]
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute(_SCHEMA_SQL)
    cur.close()
    conn.close()


# The factory caches the initialized database directory so _initialize_db
# only runs the first time; subsequent calls restore from the snapshot.
Postgresql: Any = testing.postgresql.PostgresqlFactory(  # type: ignore[name-defined,explicit-any]
    cache_initialized_db=True,
    on_initialized=_initialize_db,
)


def teardown_module() -> None:
    """Remove the cached database snapshot to avoid stale temp files."""
    Postgresql.clear_cache()


# ---------------------------------------------------------------------------
# Module-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pg_instance() -> Generator[Any, None, None]:  # type: ignore[explicit-any]
    """Start a temporary PostgreSQL instance from the cached snapshot."""
    with Postgresql() as pg:
        yield pg


@pytest.fixture(scope="module")
def extractor(pg_instance: Any) -> Generator[PostgresExtractor, None, None]:  # type: ignore[explicit-any]
    """Connect a PostgresExtractor to the test database and disconnect afterwards."""
    dsn: dict[str, Any] = pg_instance.dsn()
    ext = PostgresExtractor()
    ext.connect(
        {
            "host": dsn["host"],
            "port": dsn["port"],
            "database": dsn["database"],
            "user": dsn["user"],
            "password": "",  # trust auth — no password required
        }
    )
    yield ext
    ext.disconnect()


@pytest.fixture(scope="module")
def artifacts(extractor: PostgresExtractor) -> list[Any]:
    """Extract all artifacts from the public schema (extracted once per module)."""
    return extractor.extract_schema()


# ---------------------------------------------------------------------------
# Basic sanity
# ---------------------------------------------------------------------------


def test_language_identifier() -> None:
    """PostgresExtractor identifies itself as 'postgresql'."""
    assert PostgresExtractor().language() == "postgresql"


def test_not_connected_raises() -> None:
    """extract_schema raises RuntimeError when no connection has been made."""
    with pytest.raises(RuntimeError, match="Not connected"):
        PostgresExtractor().extract_schema()


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def test_tables_extracted(artifacts: list[Any]) -> None:
    """All three test tables appear as 'table' artifacts."""
    tables = {a.name for a in artifacts if a.type == "table"}
    assert {"category", "film", "film_category"}.issubset(tables)


def test_table_artifact_fields(artifacts: list[Any]) -> None:
    """Table artifacts carry the expected language, source_type, and module."""
    film = next(a for a in artifacts if a.type == "table" and a.name == "film")
    assert film.language == "postgresql"
    assert film.source_type == "database"
    assert film.module == "public"
    assert "table" in film.tags
    assert film.metadata["schema"] == "public"


# ---------------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------------


def test_columns_extracted(artifacts: list[Any]) -> None:
    """film table columns include expected names."""
    cols = {
        a.name for a in artifacts if a.type == "column" and a.metadata.get("table_name") == "film"
    }
    assert {"film_id", "title", "description", "rating", "length"}.issubset(cols)


def test_primary_key_column_annotated(artifacts: list[Any]) -> None:
    """category_id column is marked as a primary key."""
    col = next(
        a
        for a in artifacts
        if a.type == "column"
        and a.name == "category_id"
        and a.metadata.get("table_name") == "category"
    )
    assert col.metadata["primary_key"] is True
    assert "PRIMARY KEY" in col.constraints


def test_not_null_column(artifacts: list[Any]) -> None:
    """film.title is NOT NULL and carries the constraint."""
    col = next(
        a
        for a in artifacts
        if a.type == "column" and a.name == "title" and a.metadata.get("table_name") == "film"
    )
    assert col.metadata["nullable"] is False
    assert "NOT NULL" in col.constraints


def test_foreign_key_column_annotated(artifacts: list[Any]) -> None:
    """film_category.film_id records the referenced table.column."""
    col = next(
        a
        for a in artifacts
        if a.type == "column"
        and a.name == "film_id"
        and a.metadata.get("table_name") == "film_category"
    )
    assert col.metadata["foreign_key"] != ""
    assert "film" in col.metadata["foreign_key"]


def test_nullable_column(artifacts: list[Any]) -> None:
    """film.description is nullable."""
    col = next(
        a
        for a in artifacts
        if a.type == "column" and a.name == "description" and a.metadata.get("table_name") == "film"
    )
    assert col.metadata["nullable"] is True


# ---------------------------------------------------------------------------
# Indexes — primary key and regular
# ---------------------------------------------------------------------------


def test_primary_key_indexes_extracted(artifacts: list[Any]) -> None:
    """Primary key indexes (names ending in _pkey) are extracted as artifacts.

    Regression test: SQLAlchemy's get_indexes() silently omits PK indexes;
    PostgresExtractor must query get_pk_constraint() to capture them.
    """
    index_names = {a.name for a in artifacts if a.type == "index"}
    assert "category_pkey" in index_names, "category PK index missing"
    assert "film_pkey" in index_names, "film PK index missing"
    assert "film_category_pkey" in index_names, "film_category composite PK index missing"


def test_primary_key_index_fields(artifacts: list[Any]) -> None:
    """A _pkey index artifact is unique and carries the primary_key flag."""
    idx = next(a for a in artifacts if a.type == "index" and a.name == "film_pkey")
    assert idx.metadata["unique"] is True
    assert idx.metadata["primary_key"] is True
    assert idx.metadata["table_name"] == "film"
    assert "film_id" in idx.metadata["columns"]
    assert "primary_key" in idx.tags


def test_composite_primary_key_index(artifacts: list[Any]) -> None:
    """A composite PK index lists all constituent columns."""
    idx = next(a for a in artifacts if a.type == "index" and a.name == "film_category_pkey")
    assert set(idx.metadata["columns"]) == {"film_id", "category_id"}
    assert idx.metadata["primary_key"] is True


def test_regular_index_extracted(artifacts: list[Any]) -> None:
    """A regular (non-PK) index on film.title is extracted."""
    idx = next(
        (a for a in artifacts if a.type == "index" and a.name == "idx_film_title"),
        None,
    )
    assert idx is not None
    assert idx.metadata["unique"] is False
    assert idx.metadata.get("primary_key") is not True
    assert "title" in idx.metadata["columns"]


def test_index_artifact_fields(artifacts: list[Any]) -> None:
    """Index artifacts have the expected language, source_type, and module."""
    idx = next(a for a in artifacts if a.type == "index" and a.name == "idx_film_title")
    assert idx.language == "postgresql"
    assert idx.source_type == "database"
    assert idx.module == "public.film"


# ---------------------------------------------------------------------------
# Foreign key relationships
# ---------------------------------------------------------------------------


def test_foreign_key_relationships_extracted(artifacts: list[Any]) -> None:
    """Both film_category FK relationships are extracted."""
    rel_names = {a.name for a in artifacts if a.type == "relationship"}
    assert "fk_film_category_film" in rel_names
    assert "fk_film_category_category" in rel_names


def test_relationship_artifact_fields(artifacts: list[Any]) -> None:
    """FK relationship artifact has correct from/to table metadata."""
    rel = next(a for a in artifacts if a.name == "fk_film_category_film")
    assert rel.metadata["from_table"] == "film_category"
    assert rel.metadata["to_table"] == "film"
    assert rel.metadata["cardinality"] == "N:1"
    assert rel.language == "postgresql"
    assert "relationship" in rel.tags


def test_relationship_column_mapping(artifacts: list[Any]) -> None:
    """FK relation strings record the column-level mapping."""
    rel = next(a for a in artifacts if a.name == "fk_film_category_film")
    assert any("film_id" in r for r in rel.relations)


# ---------------------------------------------------------------------------
# Check constraints
# ---------------------------------------------------------------------------


def test_check_constraint_extracted(artifacts: list[Any]) -> None:
    """The check constraint on film.length is extracted."""
    constraints = [a for a in artifacts if a.type == "constraint"]
    chk = next((c for c in constraints if "film_length" in c.name.lower()), None)
    assert chk is not None, f"chk_film_length not found; got {[c.name for c in constraints]}"
    assert "length" in chk.description.lower() or "length" in chk.constraints[0].lower()


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------


def test_view_extracted(artifacts: list[Any]) -> None:
    """The film_in_category view is extracted."""
    views = {a.name for a in artifacts if a.type == "view"}
    assert "film_in_category" in views


def test_view_artifact_fields(artifacts: list[Any]) -> None:
    """View artifact has definition text and correct metadata."""
    view = next(a for a in artifacts if a.type == "view" and a.name == "film_in_category")
    assert view.language == "postgresql"
    assert view.module == "public"
    assert "view" in view.tags
    # The definition should reference the underlying tables
    definition: str = view.metadata.get("definition", "")
    assert "film" in definition.lower()


# ---------------------------------------------------------------------------
# Custom types (enums)
# ---------------------------------------------------------------------------


def test_enum_type_extracted(artifacts: list[Any]) -> None:
    """The mpaa_rating enum type is extracted."""
    types = {a.name for a in artifacts if a.type == "type"}
    assert "mpaa_rating" in types


def test_enum_type_fields(artifacts: list[Any]) -> None:
    """Enum type artifact records kind and enum values."""
    typ = next(a for a in artifacts if a.type == "type" and a.name == "mpaa_rating")
    assert typ.metadata["type_kind"] == "enum"
    details: str = typ.metadata.get("details", "")
    assert "G" in details
    assert "R" in details


# ---------------------------------------------------------------------------
# Sequences
# ---------------------------------------------------------------------------


def test_sequences_extracted(artifacts: list[Any]) -> None:
    """SERIAL columns generate sequences that are extracted."""
    seq_names = {a.name for a in artifacts if a.type == "sequence"}
    # SERIAL creates a sequence named <table>_<column>_seq
    assert any("category" in s for s in seq_names), f"No category sequence found; got {seq_names}"
    assert any("film" in s for s in seq_names), f"No film sequence found; got {seq_names}"


def test_sequence_artifact_fields(artifacts: list[Any]) -> None:
    """Sequence artifact contains numeric metadata."""
    seq = next(a for a in artifacts if a.type == "sequence" and "film" in a.name)
    assert seq.language == "postgresql"
    assert "sequence" in seq.tags
    assert "start_value" in seq.metadata
    assert "increment" in seq.metadata


# ---------------------------------------------------------------------------
# Triggers
# ---------------------------------------------------------------------------


def test_trigger_extracted(artifacts: list[Any]) -> None:
    """The trigger on the category table is extracted."""
    trigger_names = {a.name for a in artifacts if a.type == "trigger"}
    assert "trg_category_last_update" in trigger_names


def test_trigger_artifact_fields(artifacts: list[Any]) -> None:
    """Trigger artifact records timing, events, and table."""
    trig = next(a for a in artifacts if a.name == "trg_category_last_update")
    assert trig.metadata["table_name"] == "category"
    assert trig.metadata["timing"].upper() == "BEFORE"
    assert "UPDATE" in trig.metadata["events"].upper()
    assert "trigger" in trig.tags


# ---------------------------------------------------------------------------
# include_tables filtering
# ---------------------------------------------------------------------------


def test_include_tables_filter(extractor: PostgresExtractor) -> None:
    """include_tables restricts extraction to the requested tables."""
    arts = extractor.extract_schema(include_tables=["film"])
    table_names = {a.name for a in arts if a.type == "table"}
    assert table_names == {"film"}
    # No category or film_category columns should appear
    modules = {a.module for a in arts if a.type == "column"}
    assert all("film" in m for m in modules)


def test_include_tables_excludes_cross_schema_fk(extractor: PostgresExtractor) -> None:
    """FKs pointing to excluded tables are not emitted."""
    arts = extractor.extract_schema(include_tables=["film"])
    rel_names = {a.name for a in arts if a.type == "relationship"}
    # film has no FK pointing out to anything; film_category FKs are excluded
    assert "fk_film_category_film" not in rel_names


# ---------------------------------------------------------------------------
# Global consistency
# ---------------------------------------------------------------------------


def test_artifact_ids_are_unique(artifacts: list[Any]) -> None:
    """Every extracted artifact has a distinct ID."""
    ids = [a.id for a in artifacts]
    assert len(ids) == len(set(ids)), "Duplicate artifact IDs detected"


def test_all_artifacts_have_required_fields(artifacts: list[Any]) -> None:
    """Every artifact has non-empty id, name, type, language, module, and description."""
    for art in artifacts:
        assert art.id, f"Empty id on {art}"
        assert art.name, f"Empty name on {art}"
        assert art.type, f"Empty type on {art}"
        assert art.language == "postgresql", f"Wrong language on {art}"
        assert art.module, f"Empty module on {art}"
        assert art.description, f"Empty description on {art}"
