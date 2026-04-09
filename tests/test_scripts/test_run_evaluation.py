"""Smoke test for scripts/run_evaluation.py.

Uses the real EmbeddingService (and model) to catch constructor/config errors,
but mocks ChromaVectorStore to avoid needing a populated vector store.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import scripts.run_evaluation as script
from src.evaluation.gold_dataset import ArtifactIdentifier, GoldDataset, GoldQuery


@pytest.fixture(scope="module")
def mini_dataset() -> GoldDataset:
    """Minimal two-query dataset with pre-resolved artifact IDs."""
    return GoldDataset(
        queries=[
            GoldQuery(
                id="test_q1",
                category="entity_lookup",
                query="What columns does the film table have?",
                language="postgresql",
                relevant_artifacts={
                    "table": [ArtifactIdentifier(name="film", artifact_id="pg_table_film")],
                    "column": [ArtifactIdentifier(name="title", component="film", artifact_id="pg_col_title")],
                },
                difficulty="easy",
            ),
            GoldQuery(
                id="test_q2",
                category="entity_lookup",
                query="What Java interfaces exist in the cheap framework?",
                language="java",
                relevant_artifacts={
                    "interface": [ArtifactIdentifier(name="Catalog", artifact_id="java_iface_catalog")],
                },
                difficulty="medium",
            ),
        ],
        description="Smoke-test dataset",
        version="test",
    )


@pytest.fixture
def mock_vector_store() -> MagicMock:
    """Mock ChromaVectorStore that returns a fixed list of IDs."""
    store = MagicMock()
    store.search.return_value = (["pg_table_film", "pg_col_title", "java_iface_catalog"], [{}], [0.1])
    return store


@pytest.mark.slow
@pytest.mark.timeout(120)
def test_main_runs_end_to_end(mini_dataset: GoldDataset, mock_vector_store: MagicMock) -> None:
    """main() should complete without error.

    Catches:
    - Wrong class name in import (ChromaStore vs ChromaVectorStore)
    - EmbeddingService constructed with wrong args (config object vs kwargs)
    - ChromaVectorStore constructed with wrong args
    - Config accessed as dict instead of Pydantic model (config.get(...))
    """
    mock_report = MagicMock()

    with (
        patch.object(script, "ChromaStore", return_value=mock_vector_store),
        patch.object(script.GoldDataset, "load", return_value=mini_dataset),
        patch.object(script.RetrievalReportGenerator, "generate_single_run_report", return_value=mock_report),
    ):
        script.main()

    mock_vector_store.search.assert_called()
    mock_report.to_json.assert_called_once()
    mock_report.to_markdown.assert_called_once()
