# Performance Benchmarking Guide

This document explains how to use the performance benchmarking tools for the CHEAP RAG system.

## Quick Start

Run all benchmarks with default settings:

```bash
python scripts/benchmark_performance.py
```

## Benchmark Stages

### 1. Embedding Benchmark

Tests query embedding generation performance.

```bash
python scripts/benchmark_performance.py --stage embedding --embed-iterations 20
```

Metrics:
- Mean/median/std embedding time
- Min/max times
- Target: < 100ms per query

### 2. Retrieval Benchmark

Tests semantic search (embedding + vector store query).

```bash
python scripts/benchmark_performance.py --stage retrieval --retrieval-iterations 20
```

Metrics:
- Mean/median/std retrieval time
- Min/max times
- Target: < 100ms per query

### 3. Generation Benchmark

Tests LLM answer generation (slower, uses fewer iterations by default).

```bash
python scripts/benchmark_performance.py --stage generation --generation-iterations 5
```

Metrics:
- Mean/median/std generation time
- Min/max times
- Target: < 5s per query (Ollama), < 3s (Claude)

### 4. End-to-End Benchmark

Tests complete pipeline: embedding → retrieval → generation.

```bash
python scripts/benchmark_performance.py --stage e2e --e2e-iterations 5
```

Metrics:
- Breakdown by stage
- Total time per query
- **Target: < 10s per query** (Phase 1 success criterion)

## Configuration

Specify a custom config file:

```bash
python scripts/benchmark_performance.py --config config/claude.yaml
```

## Example Output

```
##############################################################
CHEAP RAG PERFORMANCE BENCHMARK
##############################################################

Configuration:
  Provider: ollama
  Embedding: sentence-transformers/all-mpnet-base-v2
  Device: cuda
  Vector Store: 544 artifacts

============================================================
EMBEDDING BENCHMARK
============================================================
  Query 1/10: 15.23ms
  Query 2/10: 12.45ms
  ...

Results:
  Mean:   13.45ms
  Median: 13.12ms
  Std:    2.34ms
  Min:    11.23ms
  Max:    17.89ms

============================================================
RETRIEVAL BENCHMARK
============================================================
  Query 1/10: 25.67ms
  Query 2/10: 23.45ms
  ...

Results (top_k=5):
  Mean:   24.56ms
  Median: 24.23ms
  Std:    1.89ms
  Min:    22.34ms
  Max:    28.90ms

============================================================
GENERATION BENCHMARK
============================================================
  Query 1/3: 2345.67ms (2.35s)
  Query 2/3: 2123.45ms (2.12s)
  Query 3/3: 2456.89ms (2.46s)

Results:
  Mean:   2308.67ms (2.31s)
  Median: 2345.67ms (2.35s)
  Std:    145.23ms
  Min:    2123.45ms
  Max:    2456.89ms

============================================================
END-TO-END BENCHMARK
============================================================

  Query 1/3: 'What is the sale_order table?'
    Embedding:   13.45ms
    Retrieval:   24.56ms
    Generation:  2345.67ms (2.35s)
    Total:       2383.68ms (2.38s)

  Query 2/3: 'How are orders linked to customers?'
    Embedding:   12.34ms
    Retrieval:   23.45ms
    Generation:  2123.45ms (2.12s)
    Total:       2159.24ms (2.16s)

  Query 3/3: 'What columns does the res_partner table have?'
    Embedding:   14.56ms
    Retrieval:   25.67ms
    Generation:  2456.89ms (2.46s)
    Total:       2497.12ms (2.50s)

Aggregate Results:
  Embedding:   13.45ms (avg)
  Retrieval:   24.56ms (avg)
  Generation:  2308.67ms (2.31s avg)
  Total:       2346.68ms (2.35s avg)

Performance Target: < 10s per query
  ✓ PASS: 2.35s < 10s

##############################################################
BENCHMARK COMPLETE
##############################################################
```

## Performance Targets (Phase 1)

| Stage      | Target   | Typical (RTX 4090) | Typical (CPU) |
|------------|----------|-------------------|---------------|
| Embedding  | < 100ms  | ~15ms             | ~50ms         |
| Retrieval  | < 100ms  | ~25ms             | ~40ms         |
| Generation | < 5s     | ~2-3s (Ollama)    | ~5-10s        |
| **Total**  | **< 10s**| **~2-4s**         | **~6-11s**    |

## Notes

- **Embedding**: Uses GPU if available (CUDA), otherwise CPU
- **Retrieval**: ChromaDB vector search (cosine similarity)
- **Generation**:
  - Ollama (local): 2-4s on RTX 4090 with 4-bit quantization
  - Claude API: 1-3s depending on token count
  - First query may be slower due to model loading
- **Iterations**: Use fewer iterations for generation to save time/cost

## Interpreting Results

### Good Performance
- Embedding: < 50ms (GPU), < 100ms (CPU)
- Retrieval: < 50ms
- Generation: < 3s (local), < 2s (API)
- Total: < 5s

### Acceptable Performance
- Embedding: < 100ms
- Retrieval: < 100ms
- Generation: < 5s
- Total: < 10s (meets Phase 1 target)

### Needs Optimization
- Total > 10s consistently
- Generation > 10s (check model, context length)
- Retrieval > 200ms (check index size, device)

## Troubleshooting

### Slow Embedding
- Check device (should use CUDA if available)
- Verify model loaded correctly
- Check batch size settings

### Slow Retrieval
- Check vector store size
- Verify ChromaDB persistence
- Check similarity threshold settings

### Slow Generation
- Local (Ollama): Check model quantization, GPU memory
- API (Claude): Check network, token limits
- Both: Verify context length isn't excessive

### Out of Memory
- Reduce batch size
- Use smaller model
- Check GPU memory usage
