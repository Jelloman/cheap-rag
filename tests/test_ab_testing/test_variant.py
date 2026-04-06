"""Tests for A/B testing variant configuration."""

from __future__ import annotations

import pytest

from src.ab_testing.variant import (
    BASELINE_VARIANT,
    BGE_LARGE_VARIANT,
    BGE_SMALL_VARIANT,
    E5_LARGE_VARIANT,
    VariantConfig,
)


class TestVariantConfig:
    def test_basic_creation(self):
        config = VariantConfig(
            name="test-variant",
            embedding_model="sentence-transformers/all-mpnet-base-v2",
            embedding_dimension=768,
        )
        assert config.name == "test-variant"
        assert config.embedding_model == "sentence-transformers/all-mpnet-base-v2"
        assert config.embedding_dimension == 768

    def test_defaults(self):
        config = VariantConfig(
            name="v",
            embedding_model="model",
            embedding_dimension=768,
        )
        assert config.top_k == 5
        assert config.similarity_threshold == 0.0
        assert config.metadata == {}

    def test_custom_values(self):
        config = VariantConfig(
            name="custom",
            embedding_model="some/model",
            embedding_dimension=1024,
            top_k=10,
            similarity_threshold=0.3,
            metadata={"description": "Custom variant"},
        )
        assert config.top_k == 10
        assert config.similarity_threshold == 0.3
        assert config.metadata["description"] == "Custom variant"

    def test_to_dict(self):
        config = VariantConfig(
            name="baseline",
            embedding_model="sentence-transformers/all-mpnet-base-v2",
            embedding_dimension=768,
            top_k=5,
            metadata={"description": "Test"},
        )
        d = config.to_dict()
        assert d["name"] == "baseline"
        assert d["embedding_model"] == "sentence-transformers/all-mpnet-base-v2"
        assert d["embedding_dimension"] == 768
        assert d["top_k"] == 5
        assert d["metadata"]["description"] == "Test"

    def test_to_dict_round_trip(self):
        config = VariantConfig(
            name="v1",
            embedding_model="model/x",
            embedding_dimension=512,
            top_k=3,
            similarity_threshold=0.2,
        )
        d = config.to_dict()
        assert d["name"] == config.name
        assert d["embedding_dimension"] == config.embedding_dimension
        assert d["similarity_threshold"] == config.similarity_threshold


class TestPredefinedVariants:
    def test_baseline_variant(self):
        assert BASELINE_VARIANT.name == "baseline"
        assert BASELINE_VARIANT.embedding_dimension == 768
        assert BASELINE_VARIANT.top_k == 5
        assert "all-mpnet-base-v2" in BASELINE_VARIANT.embedding_model

    def test_bge_large_variant(self):
        assert BGE_LARGE_VARIANT.name == "bge-large"
        assert BGE_LARGE_VARIANT.embedding_dimension == 1024
        assert "bge-large" in BGE_LARGE_VARIANT.embedding_model.lower()

    def test_bge_small_variant(self):
        assert BGE_SMALL_VARIANT.name == "bge-small"
        assert BGE_SMALL_VARIANT.embedding_dimension == 384
        assert "bge-small" in BGE_SMALL_VARIANT.embedding_model.lower()

    def test_e5_large_variant(self):
        assert E5_LARGE_VARIANT.name == "e5-large"
        assert E5_LARGE_VARIANT.embedding_dimension == 1024
        assert "e5" in E5_LARGE_VARIANT.embedding_model.lower()

    def test_all_variants_have_metadata(self):
        for variant in [BASELINE_VARIANT, BGE_LARGE_VARIANT, BGE_SMALL_VARIANT, E5_LARGE_VARIANT]:
            assert "description" in variant.metadata

    def test_variants_are_distinct(self):
        names = {
            BASELINE_VARIANT.name,
            BGE_LARGE_VARIANT.name,
            BGE_SMALL_VARIANT.name,
            E5_LARGE_VARIANT.name,
        }
        assert len(names) == 4  # All unique names

    def test_variant_to_dict_completeness(self):
        d = BASELINE_VARIANT.to_dict()
        required_keys = {
            "name",
            "embedding_model",
            "embedding_dimension",
            "top_k",
            "similarity_threshold",
            "metadata",
        }
        assert required_keys.issubset(d.keys())
