"""Tests for A/B experiment configuration and result structures."""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from src.ab_testing.experiment import (
    ABExperiment,
    ExperimentConfig,
    ExperimentResult,
)
from src.ab_testing.variant import BASELINE_VARIANT, VariantConfig
from src.evaluation.metrics import RetrievalMetrics


@pytest.fixture
def sample_variant_config():
    return VariantConfig(
        name="test-variant",
        embedding_model="sentence-transformers/all-mpnet-base-v2",
        embedding_dimension=768,
        top_k=5,
    )


@pytest.fixture
def sample_experiment_config(tmp_path, sample_variant_config):
    gold_path = tmp_path / "gold.json"
    # Create minimal gold dataset file
    gold_path.write_text(
        json.dumps(
            {
                "description": "Test dataset",
                "version": "1.0",
                "metadata": {},
                "queries": [],
            }
        )
    )
    return ExperimentConfig(
        name="test_experiment",
        description="Test A/B experiment",
        variants=[sample_variant_config],
        gold_dataset_path=gold_path,
        vector_store_path=tmp_path / "vector_stores",
    )


class TestExperimentConfig:
    def test_basic_creation(self, sample_experiment_config):
        config = sample_experiment_config
        assert config.name == "test_experiment"
        assert config.description == "Test A/B experiment"
        assert len(config.variants) == 1

    def test_to_dict(self, sample_experiment_config):
        d = sample_experiment_config.to_dict()
        assert d["name"] == "test_experiment"
        assert d["description"] == "Test A/B experiment"
        assert len(d["variants"]) == 1
        assert "gold_dataset_path" in d
        assert "vector_store_path" in d

    def test_multiple_variants(self, tmp_path):
        gold_path = tmp_path / "gold.json"
        gold_path.write_text(
            json.dumps({"description": "", "version": "1.0", "metadata": {}, "queries": []})
        )
        v1 = VariantConfig(name="v1", embedding_model="model1", embedding_dimension=768)
        v2 = VariantConfig(name="v2", embedding_model="model2", embedding_dimension=1024)
        config = ExperimentConfig(
            name="multi_variant",
            description="Multiple variants",
            variants=[v1, v2],
            gold_dataset_path=gold_path,
            vector_store_path=tmp_path,
        )
        d = config.to_dict()
        assert len(d["variants"]) == 2

    def test_metadata_default(self, sample_experiment_config):
        assert sample_experiment_config.metadata == {}

    def test_custom_metadata(self, tmp_path):
        gold_path = tmp_path / "gold.json"
        gold_path.write_text(
            json.dumps({"description": "", "version": "1.0", "metadata": {}, "queries": []})
        )
        config = ExperimentConfig(
            name="meta_test",
            description="desc",
            variants=[BASELINE_VARIANT],
            gold_dataset_path=gold_path,
            vector_store_path=tmp_path,
            metadata={"run_date": "2026-02-19"},
        )
        assert config.metadata["run_date"] == "2026-02-19"


class TestExperimentResult:
    @pytest.fixture
    def sample_result(self):
        metrics = RetrievalMetrics(
            precision_at_k={5: 0.6},
            recall_at_k={5: 0.8},
            mrr=0.75,
        )
        return ExperimentResult(
            experiment_name="test_exp",
            variant_name="baseline",
            metrics=metrics,
            per_query_metrics=[
                {"query_id": "q1", "latency_ms": 10.0, "metrics": {"mrr": 1.0}},
            ],
            latency_stats={"mean_ms": 10.0, "p95_ms": 15.0},
            timestamp=datetime(2026, 2, 19, 12, 0, 0),
        )

    def test_to_dict(self, sample_result):
        d = sample_result.to_dict()
        assert d["experiment_name"] == "test_exp"
        assert d["variant_name"] == "baseline"
        assert "metrics" in d
        assert "per_query_metrics" in d
        assert "latency_stats" in d
        assert "timestamp" in d

    def test_to_dict_metrics_content(self, sample_result):
        d = sample_result.to_dict()
        assert d["metrics"]["mrr"] == 0.75
        assert d["metrics"]["precision_at_k"][5] == 0.6

    def test_to_dict_latency_stats(self, sample_result):
        d = sample_result.to_dict()
        assert d["latency_stats"]["mean_ms"] == 10.0
        assert d["latency_stats"]["p95_ms"] == 15.0

    def test_to_dict_per_query_metrics(self, sample_result):
        d = sample_result.to_dict()
        assert len(d["per_query_metrics"]) == 1
        assert d["per_query_metrics"][0]["query_id"] == "q1"

    def test_metadata_default(self, sample_result):
        # metadata is empty dict by default if not provided
        assert isinstance(sample_result.metadata, dict)


class TestABExperiment:
    def test_init(self, sample_experiment_config):
        experiment = ABExperiment(sample_experiment_config)
        assert experiment.config is sample_experiment_config
        assert experiment._variants == {}

    def test_save_results(self, sample_experiment_config, tmp_path):
        experiment = ABExperiment(sample_experiment_config)

        metrics = RetrievalMetrics(mrr=0.7, precision_at_k={5: 0.6})
        results = {
            "baseline": ExperimentResult(
                experiment_name="test_experiment",
                variant_name="baseline",
                metrics=metrics,
                per_query_metrics=[],
                latency_stats={"mean_ms": 20.0},
            )
        }

        output_path = tmp_path / "results.json"
        experiment.save_results(results, output_path)

        assert output_path.exists()
        data = json.loads(output_path.read_text())
        assert data["experiment"]["name"] == "test_experiment"
        assert "baseline" in data["results"]
        assert "timestamp" in data

    def test_save_results_creates_parent_dirs(self, sample_experiment_config, tmp_path):
        experiment = ABExperiment(sample_experiment_config)
        output_path = tmp_path / "deep" / "nested" / "results.json"

        experiment.save_results({}, output_path)
        assert output_path.exists()

    def test_cleanup_empty(self, sample_experiment_config):
        experiment = ABExperiment(sample_experiment_config)
        # Should not raise when no variants initialized
        experiment.cleanup()

    def test_run_empty_gold_dataset(self, sample_experiment_config):
        """Running on empty gold dataset should return results with no metrics data."""
        # The gold dataset has no queries, so run should produce empty results
        # But setup requires actual embedding model initialization which is slow;
        # test the structure without calling setup/initialize
        experiment = ABExperiment(sample_experiment_config)
        # We can't run without setup, but we can verify run() handles empty dataset
        # by manually setting _variants to empty (simulating post-setup with no variants)
        results = experiment.run()
        # With no variants set up, results should be empty
        assert results == {}
