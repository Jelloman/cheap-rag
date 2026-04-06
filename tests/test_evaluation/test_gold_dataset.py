"""Tests for gold dataset loading and management."""

from __future__ import annotations

import json

import pytest

from src.evaluation.gold_dataset import GoldDataset, GoldQuery


@pytest.fixture
def sample_gold_queries():
    return [
        GoldQuery(
            id="q1",
            category="entity_lookup",
            query="What is sale_order?",
            language="postgresql",
            relevant_artifact_ids=["id1", "id2"],
            difficulty="easy",
        ),
        GoldQuery(
            id="q2",
            category="relationship_query",
            query="How are orders linked to customers?",
            language="postgresql",
            relevant_artifact_ids=["id3"],
            difficulty="medium",
        ),
        GoldQuery(
            id="q3",
            category="entity_lookup",
            query="Show me Java interfaces",
            language="java",
            relevant_artifact_ids=["id4", "id5"],
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


class TestGoldQuery:
    def test_to_dict(self):
        q = GoldQuery(
            id="q1",
            category="entity_lookup",
            query="Test query",
            language="postgresql",
            relevant_artifact_ids=["id1"],
            difficulty="easy",
            notes="A note",
            metadata={"source": "manual"},
        )
        d = q.to_dict()
        assert d["id"] == "q1"
        assert d["category"] == "entity_lookup"
        assert d["query"] == "Test query"
        assert d["language"] == "postgresql"
        assert d["relevant_artifact_ids"] == ["id1"]
        assert d["difficulty"] == "easy"
        assert d["notes"] == "A note"
        assert d["metadata"]["source"] == "manual"

    def test_from_dict(self):
        data = {
            "id": "q1",
            "category": "entity_lookup",
            "query": "Test",
            "language": "java",
            "relevant_artifact_ids": ["a", "b"],
            "difficulty": "hard",
            "notes": "",
            "metadata": {},
        }
        q = GoldQuery.from_dict(data)
        assert q.id == "q1"
        assert q.language == "java"
        assert q.relevant_artifact_ids == ["a", "b"]

    def test_from_dict_defaults(self):
        data = {
            "id": "q1",
            "category": "cat",
            "query": "q",
            "language": "python",
        }
        q = GoldQuery.from_dict(data)
        assert q.relevant_artifact_ids == []
        assert q.difficulty == "medium"
        assert q.notes == ""
        assert q.metadata == {}

    def test_round_trip(self):
        original = GoldQuery(
            id="q1",
            category="entity_lookup",
            query="Test",
            language="java",
            relevant_artifact_ids=["id1"],
        )
        restored = GoldQuery.from_dict(original.to_dict())
        assert restored.id == original.id
        assert restored.query == original.query
        assert restored.relevant_artifact_ids == original.relevant_artifact_ids


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
            assert original.relevant_artifact_ids == restored.relevant_artifact_ids

    def test_save_creates_valid_json(self, sample_dataset, tmp_path):
        path = tmp_path / "gold.json"
        sample_dataset.save(path)

        data = json.loads(path.read_text())
        assert "queries" in data
        assert "description" in data
        assert "version" in data
        assert len(data["queries"]) == 3

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
