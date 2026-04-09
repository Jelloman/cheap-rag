# Source Code Overview — cheap-rag (Python RAG Pipeline)

| Directory / File | Contents |
|------------------|----------|
| `src/extractors/` | Language-specific metadata extraction: Java (JavaParser via subprocess), Python (ast), TypeScript (regex), PostgreSQL and SQLite (SQLAlchemy Inspector). |
| `src/indexing/` | Unified indexing pipeline (`pipeline.py`) and ChromaDB schema definition (`schema.py`). |
| `src/embeddings/` | GPU-accelerated embedding service using sentence-transformers (`all-mpnet-base-v2`). |
| `src/vectorstore/` | ChromaDB persistent vector store (`chroma_store.py`) and optional FAISS backend (`faiss_store.py`). |
| `src/retrieval/` | Semantic search with configurable top-K (`semantic_search.py`) and metadata filtering (`filters.py`). |
| `src/generation/` | LLM answer generation with citations: prompt templates, Ollama/Anthropic providers, citation validation. |
| `src/evaluation/` | Gold dataset system, retrieval metrics (Precision@K, Recall@K, MRR, MAP, NDCG), and reporting. |
| `src/observability/` | OpenTelemetry tracing, structured logging (loguru), error tracking, and performance profiling (psutil). |
| `src/ab_testing/` | A/B experiment framework for comparing embedding models (variant definitions + experiment runner). |
| `src/api/` | FastAPI REST endpoints: `POST /api/query`, `GET /api/index/status`, `GET /api/metadata`, `GET /health`. |
| `src/config.py` | YAML configuration loading and validation; selects profile via `CONFIG_PROFILE` env var. |
| `config/` | Runtime configuration profiles: `local.yaml` (Ollama), `claude.yaml` (Claude API), `hybrid.yaml`. |
| `tests/` | 324-test suite covering all modules (unit + integration). |
| `scripts/` | CLI utilities for indexing, querying, evaluation, A/B testing, and benchmarking. |
