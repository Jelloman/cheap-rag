# CHEAP RAG - Semantic Search Over Metadata

Phase 1 implementation of the CHEAP AI Enhancement project: a RAG (Retrieval-Augmented Generation) system that enables semantic search and natural language Q&A over multi-language metadata definitions.

## Quick Start

### 1. Installation

```bash
cd cheap-rag

# Install dependencies with uv
uv pip install -e .
```

### 2. Configuration

Copy and configure your environment:

```bash
cp .env.example .env
# Edit .env with your settings
```

Configuration profiles in `config/`:
- `local.yaml` - Local-only (Ollama + sentence-transformers)
- `claude.yaml` - Claude API for generation
- `hybrid.yaml` - Hybrid routing between local and Claude

Set profile via environment variable:
```bash
export CONFIG_PROFILE=local  # or claude, hybrid
```

### 3. Index Metadata

```bash
# Index from database
python scripts/index_metadata.py --source "postgresql://user:pass@localhost/dbname"

# Index from code
python scripts/index_metadata.py --source "/path/to/java/project"

# Reset and reindex
python scripts/index_metadata.py --reset
```

### 4. Query

**CLI:**
```bash
python scripts/query_example.py "What is the sale_order table?"

# With filters
python scripts/query_example.py "What columns exist?" --language postgresql --type column

# Markdown output
python scripts/query_example.py "How are sales linked to customers?" --markdown
```

**API Server:**
```bash
# Start server
uvicorn src.api.routes:app --reload

# Or
python src/api/routes.py
```

**API Endpoints:**
- `POST /api/query` - Question answering
- `GET /api/index/status` - Index statistics
- `GET /api/metadata` - Browse metadata (basic)

Example query:
```bash
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the sale_order table?",
    "top_k": 5,
    "similarity_threshold": 0.3
  }'
```

## Architecture

```
Query Flow:
1. User query → Embedding Service (sentence-transformers)
2. Query embedding → Vector Store (ChromaDB) → Top-K similar artifacts
3. Retrieved artifacts → LLM (Qwen2.5-Coder or Claude) → Answer with citations
4. Citations validated against retrieved artifacts
5. Response with answer + sources + quality metrics
```

**Components:**
- **Extractors** - Extract metadata from databases (PostgreSQL, SQLite) and code (Java)
- **Embeddings** - Generate vector embeddings using sentence-transformers
- **Vector Store** - ChromaDB for persistent vector search
- **Retrieval** - Semantic search with filtering
- **Generation** - LLM answer generation with citations
- **API** - FastAPI endpoints

## Development Status

All Phase 1 core components complete:
- ✅ Retrieval Layer (semantic search, filters)
- ✅ LLM Answer Generation (Ollama + Claude providers)
- ✅ Citation extraction and validation
- ✅ API endpoints
- ✅ Indexing pipeline
- ✅ Test query dataset

See `../cheap-planning/TODO_PHASE_1.md` for detailed status.

## License

Part of the CHEAP metadata framework project.
