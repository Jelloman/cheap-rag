"""Performance benchmarking for CHEAP RAG pipeline."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

from src.config import load_config
from src.embeddings.service import EmbeddingService
from src.generation.generator import AnthropicProvider, Generator, OllamaProvider
from src.retrieval.semantic_search import SemanticSearch
from src.vectorstore.chroma_store import ChromaVectorStore


class PerformanceBenchmark:
    """Performance benchmarking for RAG pipeline stages."""

    def __init__(self, config_path: str | None = None):
        """Initialize benchmark with configuration.

        Args:
            config_path: Path to config file (optional).
        """
        print("Loading configuration...")
        self.config = load_config(config_path) if config_path else load_config()

        print(f"Provider: {self.config.llm.provider}")
        print(f"Embedding model: {self.config.embedding.model_name}")

        # Initialize services
        print("\nInitializing services...")
        self.embedding_service = EmbeddingService(
            model_name=self.config.embedding.model_name,
            device=self.config.embedding.device,
            cache_dir=self.config.embedding.cache_dir,
        )

        self.vector_store = ChromaVectorStore(
            persist_directory=self.config.vectorstore.persist_directory,
            collection_name=self.config.vectorstore.collection_name,
        )

        self.semantic_search = SemanticSearch(
            embedding_service=self.embedding_service,
            vector_store=self.vector_store,
        )

        # Initialize LLM provider
        if self.config.llm.provider == "ollama":
            if self.config.llm.ollama is None:
                raise ValueError("Ollama config required")
            provider = OllamaProvider(self.config.llm.ollama)
        elif self.config.llm.provider == "anthropic":
            if self.config.llm.anthropic is None:
                raise ValueError("Anthropic config required")
            from src.config import get_anthropic_api_key

            api_key = get_anthropic_api_key()
            provider = AnthropicProvider(self.config.llm.anthropic, api_key)
        else:
            raise ValueError(f"Unknown provider: {self.config.llm.provider}")

        self.generator = Generator(provider)

        # Test queries
        self.test_queries = [
            "What is the sale_order table?",
            "How are orders linked to customers?",
            "What columns does the res_partner table have?",
            "What indexes exist on the sale_order table?",
            "How do I find all orders for a partner?",
        ]

        print(f"Services initialized. Vector store has {self.vector_store.count()} artifacts.")

    def benchmark_embedding(self, iterations: int = 10) -> dict[str, float]:
        """Benchmark embedding generation.

        Args:
            iterations: Number of iterations.

        Returns:
            Dict with timing statistics.
        """
        print(f"\n{'='*60}")
        print("EMBEDDING BENCHMARK")
        print(f"{'='*60}")

        times = []

        for i, query in enumerate(self.test_queries * (iterations // len(self.test_queries) + 1)):
            if i >= iterations:
                break

            start = time.perf_counter()
            _ = self.embedding_service.embed_text(query)
            elapsed = (time.perf_counter() - start) * 1000

            times.append(elapsed)
            print(f"  Query {i+1}/{iterations}: {elapsed:.2f}ms")

        stats = {
            "mean_ms": mean(times),
            "median_ms": median(times),
            "std_ms": stdev(times) if len(times) > 1 else 0,
            "min_ms": min(times),
            "max_ms": max(times),
            "iterations": iterations,
        }

        print(f"\nResults:")
        print(f"  Mean:   {stats['mean_ms']:.2f}ms")
        print(f"  Median: {stats['median_ms']:.2f}ms")
        print(f"  Std:    {stats['std_ms']:.2f}ms")
        print(f"  Min:    {stats['min_ms']:.2f}ms")
        print(f"  Max:    {stats['max_ms']:.2f}ms")

        return stats

    def benchmark_retrieval(self, iterations: int = 10, top_k: int = 5) -> dict[str, float]:
        """Benchmark semantic search retrieval.

        Args:
            iterations: Number of iterations.
            top_k: Number of results to retrieve.

        Returns:
            Dict with timing statistics.
        """
        print(f"\n{'='*60}")
        print("RETRIEVAL BENCHMARK")
        print(f"{'='*60}")

        times = []

        for i, query in enumerate(self.test_queries * (iterations // len(self.test_queries) + 1)):
            if i >= iterations:
                break

            start = time.perf_counter()
            _ = self.semantic_search.search(query, top_k=top_k)
            elapsed = (time.perf_counter() - start) * 1000

            times.append(elapsed)
            print(f"  Query {i+1}/{iterations}: {elapsed:.2f}ms")

        stats = {
            "mean_ms": mean(times),
            "median_ms": median(times),
            "std_ms": stdev(times) if len(times) > 1 else 0,
            "min_ms": min(times),
            "max_ms": max(times),
            "iterations": iterations,
            "top_k": top_k,
        }

        print(f"\nResults (top_k={top_k}):")
        print(f"  Mean:   {stats['mean_ms']:.2f}ms")
        print(f"  Median: {stats['median_ms']:.2f}ms")
        print(f"  Std:    {stats['std_ms']:.2f}ms")
        print(f"  Min:    {stats['min_ms']:.2f}ms")
        print(f"  Max:    {stats['max_ms']:.2f}ms")

        return stats

    def benchmark_generation(self, iterations: int = 3) -> dict[str, float]:
        """Benchmark LLM answer generation.

        Args:
            iterations: Number of iterations (fewer due to cost/time).

        Returns:
            Dict with timing statistics.
        """
        print(f"\n{'='*60}")
        print("GENERATION BENCHMARK")
        print(f"{'='*60}")

        times = []

        # Use fewer iterations for generation (it's slower and potentially costly)
        test_queries = self.test_queries[:iterations]

        for i, query in enumerate(test_queries):
            # Get search results first
            search_results = self.semantic_search.search(query, top_k=5)

            if not search_results.results:
                print(f"  Query {i+1}/{iterations}: SKIPPED (no results)")
                continue

            start = time.perf_counter()
            _ = self.generator.generate_answer(query, search_results.results)
            elapsed = (time.perf_counter() - start) * 1000

            times.append(elapsed)
            print(f"  Query {i+1}/{iterations}: {elapsed:.2f}ms")

        if not times:
            return {
                "mean_ms": 0,
                "median_ms": 0,
                "std_ms": 0,
                "min_ms": 0,
                "max_ms": 0,
                "iterations": 0,
            }

        stats = {
            "mean_ms": mean(times),
            "median_ms": median(times),
            "std_ms": stdev(times) if len(times) > 1 else 0,
            "min_ms": min(times),
            "max_ms": max(times),
            "iterations": len(times),
        }

        print(f"\nResults:")
        print(f"  Mean:   {stats['mean_ms']:.2f}ms ({stats['mean_ms']/1000:.2f}s)")
        print(f"  Median: {stats['median_ms']:.2f}ms ({stats['median_ms']/1000:.2f}s)")
        print(f"  Std:    {stats['std_ms']:.2f}ms")
        print(f"  Min:    {stats['min_ms']:.2f}ms")
        print(f"  Max:    {stats['max_ms']:.2f}ms")

        return stats

    def benchmark_end_to_end(self, iterations: int = 3) -> dict[str, Any]:
        """Benchmark complete pipeline end-to-end.

        Args:
            iterations: Number of iterations.

        Returns:
            Dict with timing statistics and breakdown.
        """
        print(f"\n{'='*60}")
        print("END-TO-END BENCHMARK")
        print(f"{'='*60}")

        results = []

        test_queries = self.test_queries[:iterations]

        for i, query in enumerate(test_queries):
            print(f"\n  Query {i+1}/{iterations}: '{query}'")

            # Embedding
            embed_start = time.perf_counter()
            _ = self.embedding_service.embed_text(query)
            embed_time = (time.perf_counter() - embed_start) * 1000
            print(f"    Embedding:   {embed_time:.2f}ms")

            # Retrieval
            retrieval_start = time.perf_counter()
            search_results = self.semantic_search.search(query, top_k=5)
            retrieval_time = (time.perf_counter() - retrieval_start) * 1000
            print(f"    Retrieval:   {retrieval_time:.2f}ms")

            if not search_results.results:
                print("    Generation:  SKIPPED (no results)")
                continue

            # Generation
            generation_start = time.perf_counter()
            _ = self.generator.generate_answer(query, search_results.results)
            generation_time = (time.perf_counter() - generation_start) * 1000
            print(f"    Generation:  {generation_time:.2f}ms ({generation_time/1000:.2f}s)")

            total_time = embed_time + retrieval_time + generation_time
            print(f"    Total:       {total_time:.2f}ms ({total_time/1000:.2f}s)")

            results.append({
                "embedding_ms": embed_time,
                "retrieval_ms": retrieval_time,
                "generation_ms": generation_time,
                "total_ms": total_time,
            })

        if not results:
            return {"iterations": 0}

        # Aggregate statistics
        stats = {
            "iterations": len(results),
            "embedding": {
                "mean_ms": mean([r["embedding_ms"] for r in results]),
                "median_ms": median([r["embedding_ms"] for r in results]),
            },
            "retrieval": {
                "mean_ms": mean([r["retrieval_ms"] for r in results]),
                "median_ms": median([r["retrieval_ms"] for r in results]),
            },
            "generation": {
                "mean_ms": mean([r["generation_ms"] for r in results]),
                "median_ms": median([r["generation_ms"] for r in results]),
            },
            "total": {
                "mean_ms": mean([r["total_ms"] for r in results]),
                "median_ms": median([r["total_ms"] for r in results]),
            },
        }

        print(f"\nAggregate Results:")
        print(f"  Embedding:   {stats['embedding']['mean_ms']:.2f}ms (avg)")
        print(f"  Retrieval:   {stats['retrieval']['mean_ms']:.2f}ms (avg)")
        print(f"  Generation:  {stats['generation']['mean_ms']:.2f}ms ({stats['generation']['mean_ms']/1000:.2f}s avg)")
        print(f"  Total:       {stats['total']['mean_ms']:.2f}ms ({stats['total']['mean_ms']/1000:.2f}s avg)")

        # Check performance target (< 10s)
        avg_total_s = stats["total"]["mean_ms"] / 1000
        print(f"\nPerformance Target: < 10s per query")
        if avg_total_s < 10:
            print(f"  ✓ PASS: {avg_total_s:.2f}s < 10s")
        else:
            print(f"  ✗ FAIL: {avg_total_s:.2f}s >= 10s")

        return stats

    def run_all_benchmarks(
        self,
        embed_iterations: int = 10,
        retrieval_iterations: int = 10,
        generation_iterations: int = 3,
        e2e_iterations: int = 3,
    ) -> dict[str, Any]:
        """Run all benchmarks.

        Args:
            embed_iterations: Embedding iterations.
            retrieval_iterations: Retrieval iterations.
            generation_iterations: Generation iterations.
            e2e_iterations: End-to-end iterations.

        Returns:
            Dict with all benchmark results.
        """
        print(f"\n{'#'*60}")
        print("CHEAP RAG PERFORMANCE BENCHMARK")
        print(f"{'#'*60}")
        print(f"\nConfiguration:")
        print(f"  Provider: {self.config.llm.provider}")
        print(f"  Embedding: {self.config.embedding.model_name}")
        print(f"  Device: {self.config.embedding.device}")
        print(f"  Vector Store: {self.vector_store.count()} artifacts")

        results = {
            "embedding": self.benchmark_embedding(embed_iterations),
            "retrieval": self.benchmark_retrieval(retrieval_iterations),
            "generation": self.benchmark_generation(generation_iterations),
            "end_to_end": self.benchmark_end_to_end(e2e_iterations),
        }

        print(f"\n{'#'*60}")
        print("BENCHMARK COMPLETE")
        print(f"{'#'*60}")

        return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Benchmark CHEAP RAG performance")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--embed-iterations", type=int, default=10, help="Embedding iterations")
    parser.add_argument(
        "--retrieval-iterations", type=int, default=10, help="Retrieval iterations"
    )
    parser.add_argument(
        "--generation-iterations", type=int, default=3, help="Generation iterations"
    )
    parser.add_argument("--e2e-iterations", type=int, default=3, help="End-to-end iterations")
    parser.add_argument(
        "--stage",
        type=str,
        choices=["embedding", "retrieval", "generation", "e2e", "all"],
        default="all",
        help="Which stage to benchmark",
    )

    args = parser.parse_args()

    benchmark = PerformanceBenchmark(args.config)

    if args.stage == "embedding":
        benchmark.benchmark_embedding(args.embed_iterations)
    elif args.stage == "retrieval":
        benchmark.benchmark_retrieval(args.retrieval_iterations)
    elif args.stage == "generation":
        benchmark.benchmark_generation(args.generation_iterations)
    elif args.stage == "e2e":
        benchmark.benchmark_end_to_end(args.e2e_iterations)
    else:  # all
        benchmark.run_all_benchmarks(
            embed_iterations=args.embed_iterations,
            retrieval_iterations=args.retrieval_iterations,
            generation_iterations=args.generation_iterations,
            e2e_iterations=args.e2e_iterations,
        )


if __name__ == "__main__":
    main()
