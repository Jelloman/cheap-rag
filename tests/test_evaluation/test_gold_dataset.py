"""Tests for gold dataset loading and management."""

from __future__ import annotations

import json

import pytest

from src.evaluation.gold_dataset import ArtifactIdentifier, GoldDataset, GoldQuery


def _ident(artifact_id: str, name: str = "", component: str | None = None) -> ArtifactIdentifier:
    """Create an ArtifactIdentifier for use in tests."""
    return ArtifactIdentifier(name=name, component=component, artifact_id=artifact_id)


@pytest.fixture
def sample_gold_queries():
    return [
        GoldQuery(
            id="q1",
            category="entity_lookup",
            query="What is sale_order?",
            language="postgresql",
            relevant_artifacts={"table": [_ident("id1", "sale_order")]},
            difficulty="easy",
        ),
        GoldQuery(
            id="q2",
            category="relationship_query",
            query="How are orders linked to customers?",
            language="postgresql",
            relevant_artifacts={"relationship": [_ident("id3", "sale_order_customer_fk")]},
            difficulty="medium",
        ),
        GoldQuery(
            id="q3",
            category="entity_lookup",
            query="Show me Java interfaces",
            language="java",
            relevant_artifacts={
                "interface": [_ident("id4", "Catalog"), _ident("id5", "Repository")]
            },
            difficulty="hard",
        ),
    ]


@pytest.fixture
def sample_dataset(sample_gold_queries):
    return GoldDataset(
        queries=sample_gold_queries,
        description="Test dataset",
        version="1.0",
    )


class TestArtifactIdentifier:
    def test_to_dict_minimal(self):
        ident = ArtifactIdentifier(name="film")
        assert ident.to_dict() == {"name": "film"}

    def test_to_dict_with_component(self):
        ident = ArtifactIdentifier(name="length", component="film")
        assert ident.to_dict() == {"name": "length", "component": "film"}

    def test_to_dict_full(self):
        ident = ArtifactIdentifier(name="length", component="film", artifact_id="pg_col_abc")
        assert ident.to_dict() == {
            "name": "length",
            "component": "film",
            "artifact_id": "pg_col_abc",
        }

    def test_from_dict_minimal(self):
        ident = ArtifactIdentifier.from_dict({"name": "film"})
        assert ident.name == "film"
        assert ident.component is None
        assert ident.artifact_id is None

    def test_from_dict_full(self):
        ident = ArtifactIdentifier.from_dict(
            {"name": "length", "component": "film", "artifact_id": "pg_col_abc"}
        )
        assert ident.name == "length"
        assert ident.component == "film"
        assert ident.artifact_id == "pg_col_abc"

    def test_round_trip(self):
        original = ArtifactIdentifier(name="title", component="film", artifact_id="pg_col_xyz")
        restored = ArtifactIdentifier.from_dict(original.to_dict())
        assert restored == original


class TestGoldQuery:
    def test_to_dict(self):
        q = GoldQuery(
            id="q1",
            category="entity_lookup",
            query="Test query",
            language="postgresql",
            relevant_artifacts={
                "table": [ArtifactIdentifier(name="film", artifact_id="pg_table_abc")],
                "column": [
                    ArtifactIdentifier(name="title", component="film", artifact_id="pg_col_xyz")
                ],
            },
            difficulty="easy",
            notes="A note",
            metadata={"source": "manual"},
        )
        d = q.to_dict()
        assert d["id"] == "q1"
        assert d["language"] == "postgresql"
        assert "relevant_artifacts" in d
        assert "table" in d["relevant_artifacts"]
        assert d["relevant_artifacts"]["table"] == [{"name": "film", "artifact_id": "pg_table_abc"}]
        assert d["relevant_artifacts"]["column"] == [
            {"name": "title", "component": "film", "artifact_id": "pg_col_xyz"}
        ]
        assert "relevant_artifact_ids" not in d

    def test_from_dict_new_format(self):
        data = {
            "id": "q1",
            "category": "entity_lookup",
            "query": "Test",
            "language": "java",
            "relevant_artifacts": {
                "class": [{"name": "Catalog", "artifact_id": "java_class_abc"}],
                "field": [{"name": "id", "component": "Catalog", "artifact_id": "java_field_xyz"}],
            },
            "difficulty": "hard",
            "notes": "",
            "metadata": {},
        }
        q = GoldQuery.from_dict(data)
        assert q.id == "q1"
        assert q.language == "java"
        assert set(q.relevant_artifacts.keys()) == {"class", "field"}
        assert q.relevant_artifacts["class"][0].name == "Catalog"
        assert q.relevant_artifacts["field"][0].component == "Catalog"
        assert q.relevant_ids == {"java_class_abc", "java_field_xyz"}

    def test_from_dict_defaults(self):
        data = {"id": "q1", "category": "cat", "query": "q", "language": "python"}
        q = GoldQuery.from_dict(data)
        assert q.relevant_artifacts == {}
        assert q.relevant_ids == set()
        assert q.difficulty == "medium"
        assert q.notes == ""
        assert q.metadata == {}

    def test_relevant_ids_only_resolved(self):
        """relevant_ids skips identifiers without artifact_id."""
        q = GoldQuery(
            id="q1",
            category="c",
            query="q",
            language="java",
            relevant_artifacts={
                "class": [
                    ArtifactIdentifier(name="Foo", artifact_id="java_abc"),
                    ArtifactIdentifier(name="Bar"),  # no artifact_id — hand-authored
                ]
            },
        )
        assert q.relevant_ids == {"java_abc"}

    def test_round_trip(self):
        original = GoldQuery(
            id="q1",
            category="entity_lookup",
            query="Test",
            language="java",
            relevant_artifacts={
                "class": [ArtifactIdentifier(name="MyClass", artifact_id="java_abc")],
            },
        )
        restored = GoldQuery.from_dict(original.to_dict())
        assert restored.id == original.id
        assert restored.query == original.query
        assert restored.relevant_ids == original.relevant_ids
        assert restored.relevant_artifacts["class"][0].name == "MyClass"


class TestGoldDataset:
    def test_len(self, sample_dataset):
        assert len(sample_dataset) == 3

    def test_iter(self, sample_dataset):
        ids = [q.id for q in sample_dataset]
        assert ids == ["q1", "q2", "q3"]

    def test_filter_by_category(self, sample_dataset):
        entity_queries = sample_dataset.filter_by_category("entity_lookup")
        assert len(entity_queries) == 2
        assert all(q.category == "entity_lookup" for q in entity_queries)

    def test_filter_by_language(self, sample_dataset):
        pg_queries = sample_dataset.filter_by_language("postgresql")
        assert len(pg_queries) == 2
        assert all(q.language == "postgresql" for q in pg_queries)

    def test_filter_by_difficulty(self, sample_dataset):
        easy_queries = sample_dataset.filter_by_difficulty("easy")
        assert len(easy_queries) == 1
        assert easy_queries[0].id == "q1"

    def test_filter_by_nonexistent_category(self, sample_dataset):
        assert sample_dataset.filter_by_category("nonexistent") == []

    def test_save_and_load(self, sample_dataset, tmp_path):
        path = tmp_path / "gold.json"
        sample_dataset.save(path)

        loaded = GoldDataset.load(path)
        assert len(loaded) == len(sample_dataset)
        assert loaded.description == sample_dataset.description
        assert loaded.version == sample_dataset.version

        for original, restored in zip(sample_dataset, loaded):
            assert original.id == restored.id
            assert original.query == restored.query
            assert original.relevant_ids == restored.relevant_ids

    def test_save_uses_new_format(self, sample_dataset, tmp_path):
        path = tmp_path / "gold.json"
        sample_dataset.save(path)

        raw = json.loads(path.read_text())
        assert "queries" in raw
        for q in raw["queries"]:
            assert "relevant_artifacts" in q
            assert "relevant_artifact_ids" not in q

    def test_load_preserves_metadata(self, tmp_path):
        dataset = GoldDataset(
            queries=[],
            description="Custom desc",
            version="2.0",
            metadata={"generated_from": "test"},
        )
        path = tmp_path / "gold.json"
        dataset.save(path)

        loaded = GoldDataset.load(path)
        assert loaded.description == "Custom desc"
        assert loaded.version == "2.0"
