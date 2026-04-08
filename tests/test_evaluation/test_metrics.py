"""Tests for retrieval evaluation metrics."""

from __future__ import annotations

import pytest

from src.evaluation.gold_dataset import GoldQuery
from src.evaluation.metrics import (
    RetrievalMetrics,
    aggregate_retrieval_metrics,
    calculate_average_precision,
    calculate_mrr,
    calculate_ndcg,
    calculate_precision_at_k,
    calculate_recall_at_k,
    evaluate_retrieval,
)


class TestPrecisionAtK:
    def test_perfect_precision(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        assert calculate_precision_at_k(retrieved, relevant, 3) == pytest.approx(1.0)

    def test_zero_precision(self):
        retrieved = ["x", "y", "z"]
        relevant = {"a", "b", "c"}
        assert calculate_precision_at_k(retrieved, relevant, 3) == pytest.approx(0.0)

    def test_partial_precision(self):
        retrieved = ["a", "x", "b"]
        relevant = {"a", "b"}
        assert calculate_precision_at_k(retrieved, relevant, 3) == pytest.approx(2 / 3)

    def test_k_larger_than_results(self):
        retrieved = ["a", "b"]
        relevant = {"a", "b", "c"}
        assert calculate_precision_at_k(retrieved, relevant, 5) == pytest.approx(2 / 5)

    def test_empty_retrieved(self):
        assert calculate_precision_at_k([], {"a"}, 5) == 0.0

    def test_k_zero(self):
        assert calculate_precision_at_k(["a"], {"a"}, 0) == 0.0

    def test_precision_at_1(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a"}
        assert calculate_precision_at_k(retrieved, relevant, 1) == pytest.approx(1.0)

    def test_precision_at_1_miss(self):
        retrieved = ["b", "a", "c"]
        relevant = {"a"}
        assert calculate_precision_at_k(retrieved, relevant, 1) == pytest.approx(0.0)


class TestRecallAtK:
    def test_perfect_recall(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        assert calculate_recall_at_k(retrieved, relevant, 3) == pytest.approx(1.0)

    def test_zero_recall(self):
        retrieved = ["x", "y", "z"]
        relevant = {"a", "b"}
        assert calculate_recall_at_k(retrieved, relevant, 3) == pytest.approx(0.0)

    def test_partial_recall(self):
        retrieved = ["a", "x", "y", "b", "z"]
        relevant = {"a", "b", "c"}
        assert calculate_recall_at_k(retrieved, relevant, 5) == pytest.approx(2 / 3)

    def test_empty_relevant(self):
        assert calculate_recall_at_k(["a"], set(), 5) == 0.0

    def test_k_zero(self):
        assert calculate_recall_at_k(["a"], {"a"}, 0) == 0.0

    def test_recall_truncates_at_k(self):
        # Only top-2 considered, "b" is at position 3
        retrieved = ["a", "x", "b"]
        relevant = {"a", "b"}
        assert calculate_recall_at_k(retrieved, relevant, 2) == pytest.approx(0.5)


class TestMRR:
    def test_first_result_relevant(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a"}
        assert calculate_mrr(retrieved, relevant) == pytest.approx(1.0)

    def test_second_result_relevant(self):
        retrieved = ["x", "a", "b"]
        relevant = {"a"}
        assert calculate_mrr(retrieved, relevant) == pytest.approx(0.5)

    def test_third_result_relevant(self):
        retrieved = ["x", "y", "a"]
        relevant = {"a"}
        assert calculate_mrr(retrieved, relevant) == pytest.approx(1 / 3)

    def test_no_relevant_results(self):
        retrieved = ["x", "y", "z"]
        relevant = {"a"}
        assert calculate_mrr(retrieved, relevant) == pytest.approx(0.0)

    def test_empty_retrieved(self):
        assert calculate_mrr([], {"a"}) == pytest.approx(0.0)

    def test_uses_first_relevant(self):
        # MRR uses rank of FIRST relevant item, ignores subsequent
        retrieved = ["x", "a", "b"]
        relevant = {"a", "b"}
        assert calculate_mrr(retrieved, relevant) == pytest.approx(0.5)


class TestAveragePrecision:
    def test_all_relevant(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        ap = calculate_average_precision(retrieved, relevant)
        assert ap == pytest.approx(1.0)

    def test_no_relevant(self):
        retrieved = ["x", "y", "z"]
        relevant = {"a"}
        assert calculate_average_precision(retrieved, relevant) == pytest.approx(0.0)

    def test_empty_relevant(self):
        assert calculate_average_precision(["a", "b"], set()) == pytest.approx(0.0)

    def test_partial_ap(self):
        # a@1 relevant (P@1=1.0), b@3 relevant (P@3=2/3), c not in top-3
        retrieved = ["a", "x", "b"]
        relevant = {"a", "b"}
        # AP = (1.0 + 2/3) / 2 = 0.833...
        ap = calculate_average_precision(retrieved, relevant)
        assert ap == pytest.approx((1.0 + 2 / 3) / 2)


class TestNDCG:
    def test_perfect_ndcg(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        ndcg = calculate_ndcg(retrieved, relevant, k=3)
        assert ndcg == pytest.approx(1.0)

    def test_zero_ndcg(self):
        retrieved = ["x", "y", "z"]
        relevant = {"a", "b"}
        assert calculate_ndcg(retrieved, relevant) == pytest.approx(0.0)

    def test_empty_relevant(self):
        assert calculate_ndcg(["a"], set()) == pytest.approx(0.0)

    def test_ndcg_with_k(self):
        retrieved = ["a", "x", "b"]
        relevant = {"a", "b"}
        ndcg = calculate_ndcg(retrieved, relevant, k=3)
        assert 0.0 < ndcg < 1.0

    def test_higher_rank_better(self):
        # a at rank 1 vs b at rank 1 — both perfect, so both == 1.0
        retrieved_good = ["a", "b"]
        retrieved_bad = ["x", "a", "b"]
        relevant = {"a", "b"}
        ndcg_good = calculate_ndcg(retrieved_good, relevant, k=2)
        ndcg_bad = calculate_ndcg(retrieved_bad, relevant, k=2)
        assert ndcg_good >= ndcg_bad


class TestEvaluateRetrieval:
    @pytest.fixture
    def gold_query(self):
        from src.evaluation.gold_dataset import ArtifactIdentifier

        return GoldQuery(
            id="q1",
            category="entity_lookup",
            query="What is sale_order?",
            language="postgresql",
            relevant_artifacts={
                "table": [
                    ArtifactIdentifier(name="id1", artifact_id="id1"),
                    ArtifactIdentifier(name="id2", artifact_id="id2"),
                    ArtifactIdentifier(name="id3", artifact_id="id3"),
                ]
            },
            difficulty="easy",
        )

    def test_evaluate_retrieval_perfect(self, gold_query):
        retrieved = ["id1", "id2", "id3"]
        metrics = evaluate_retrieval(gold_query, retrieved)

        assert metrics.precision_at_k[5] == pytest.approx(3 / 5)
        assert metrics.recall_at_k[3] == pytest.approx(1.0)
        assert metrics.mrr == pytest.approx(1.0)

    def test_evaluate_retrieval_empty(self, gold_query):
        metrics = evaluate_retrieval(gold_query, [])
        assert metrics.mrr == pytest.approx(0.0)
        assert all(p == 0.0 for p in metrics.precision_at_k.values())

    def test_evaluate_retrieval_default_k_values(self, gold_query):
        metrics = evaluate_retrieval(gold_query, ["id1"])
        assert 1 in metrics.precision_at_k
        assert 3 in metrics.precision_at_k
        assert 5 in metrics.precision_at_k
        assert 10 in metrics.precision_at_k

    def test_evaluate_retrieval_custom_k_values(self, gold_query):
        metrics = evaluate_retrieval(gold_query, ["id1"], k_values=[2, 4])
        assert 2 in metrics.precision_at_k
        assert 4 in metrics.precision_at_k
        assert 1 not in metrics.precision_at_k

    def test_evaluate_retrieval_includes_metadata(self, gold_query):
        metrics = evaluate_retrieval(gold_query, ["id1"])
        assert metrics.metadata["query_id"] == "q1"
        assert metrics.metadata["num_relevant"] == 3


class TestAggregateRetrievalMetrics:
    def test_aggregate_empty_list(self):
        result = aggregate_retrieval_metrics([])
        assert isinstance(result, RetrievalMetrics)

    def test_aggregate_single(self):
        m = RetrievalMetrics(
            precision_at_k={5: 0.6},
            recall_at_k={5: 0.8},
            mrr=0.75,
        )
        result = aggregate_retrieval_metrics([m])
        assert result.mrr == pytest.approx(0.75)
        assert result.precision_at_k[5] == pytest.approx(0.6)

    def test_aggregate_mean(self):
        m1 = RetrievalMetrics(precision_at_k={5: 0.4}, recall_at_k={5: 0.6}, mrr=0.5)
        m2 = RetrievalMetrics(precision_at_k={5: 0.6}, recall_at_k={5: 0.8}, mrr=0.7)
        result = aggregate_retrieval_metrics([m1, m2])
        assert result.precision_at_k[5] == pytest.approx(0.5)
        assert result.recall_at_k[5] == pytest.approx(0.7)
        assert result.mrr == pytest.approx(0.6)

    def test_aggregate_metadata(self):
        m1 = RetrievalMetrics(mrr=0.5)
        m2 = RetrievalMetrics(mrr=0.7)
        result = aggregate_retrieval_metrics([m1, m2])
        assert result.metadata["num_queries"] == 2
        assert result.metadata["aggregation"] == "mean"


class TestRetrievalMetrics:
    def test_to_dict(self):
        metrics = RetrievalMetrics(
            precision_at_k={5: 0.6},
            recall_at_k={5: 0.8},
            mrr=0.75,
            map_score=0.65,
            ndcg=0.7,
        )
        d = metrics.to_dict()
        assert d["mrr"] == 0.75
        assert d["map"] == 0.65
        assert d["ndcg"] == 0.7
        assert d["precision_at_k"][5] == 0.6
