# CHEAP RAG - Metadata Semantic Search

Phase 1 implementation of the CHEAP AI Enhancement project: Core RAG pipeline with embeddings and vector search over multi-language metadata.

## Overview

This system provides semantic question-answering over CHEAP metadata definitions across Java, TypeScript, Python, and Rust. It combines:

- **Metadata extraction** from source code annotations and types
- **Semantic embeddings** using sentence-transformers
- **Vector search** with ChromaDB
- **LLM-powered generation** with citations using local models or Claude API

## Project Status

**Phase:** 1 - Core RAG + Embeddings + Vector Search
**Status:** Initial setup
**Timeline:** Weeks 1-2

## Technology Stack

- **Language:** Python 3.14
- **Embeddings:** sentence-transformers/all-mpnet-base-v2 (local)
- **Vector Store:** ChromaDB (local persistence)
- **LLM (Default):** Qwen2.5-Coder-7B-Instruct via Ollama
- **LLM (Alternate):** Claude Sonnet 4.5 / Haiku 4.5 via API
- **API Framework:** FastAPI

See [TECH_STACK_DECISIONS.md](../cheap-planning/TECH_STACK_DECISIONS.md) for detailed rationale.

## Quick Start

### Prerequisites

- Python 3.11+
- CUDA-capable GPU (recommended: 8GB+ VRAM)
- [Ollama](https://ollama.ai) installed (for local LLM)

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download Ollama model
ollama pull qwen2.5-coder:7b-instruct-q4_K_M

# Download embedding model (automatic on first run)
python scripts/download_models.py
```

### Configuration

Three configuration profiles are available:

1. **Local (Default)**: `config/local.yaml` - All processing on local GPU
2. **Claude**: `config/claude.yaml` - Claude API for generation
3. **Hybrid**: `config/hybrid.yaml` - Configurable mix

Copy `.env.example` to `.env` and configure:

```bash
# For Claude API mode
ANTHROPIC_API_KEY=your_api_key_here

# Configuration profile
CONFIG_PROFILE=local  # or claude, or hybrid
```

### Usage

#### 1. Index Metadata

```bash
# Extract and index metadata from CHEAP projects
python scripts/index_metadata.py \
  --source ../cheap-core/src/main/java \
  --source ../cheap-ts/src \
  --source ../cheap-py/src
```

#### 2. Query the System

```bash
# Interactive query
python scripts/query_example.py

# Or programmatically
from src.generation.generator import MetadataRAG

rag = MetadataRAG(config_path="config/local.yaml")
answer = rag.answer_question("What is the Catalog interface in Java?")
print(answer.answer_text)
print(answer.sources)
```

#### 3. Start API Server

```bash
# Run FastAPI server
uvicorn src.api.routes:app --reload --port 8000

# Query via HTTP
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What constraints are defined for User fields?"}'
```

## Project Structure

```
cheap-rag/
├── config/               # Configuration files
│   ├── local.yaml        # Local LLM configuration (default)
│   ├── claude.yaml       # Claude API configuration
│   └── hybrid.yaml       # Hybrid mode configuration
├── src/
│   ├── extractors/       # Metadata extraction from source code
│   ├── indexing/         # Indexing pipeline and schema
│   ├── embeddings/       # Embedding generation service
│   ├── vectorstore/      # ChromaDB integration
│   ├── retrieval/        # Semantic search and filtering
│   ├── generation/       # LLM-powered answer generation
│   └── api/              # FastAPI endpoints
├── tests/                # Test suites
├── data/
│   ├── metadata/         # Extracted metadata artifacts (JSON)
│   └── vector_db/        # ChromaDB persistence
├── scripts/
│   ├── download_models.py
│   ├── index_metadata.py
│   └── query_example.py
├── requirements.txt      # Python dependencies
└── README.md
```

## Development

### Running Tests

```bash
# All tests
pytest

# Specific module
pytest tests/test_extractors/

# With coverage
pytest --cov=src --cov-report=html
```

### Configuration Modes

#### Local Mode (Default)
- Zero API costs
- Runs entirely on local GPU
- Good for development and high query volume

#### Claude Mode
- Highest quality answers
- API costs apply
- Best for quality benchmarking

#### Hybrid Mode
- Use local for most queries
- Fallback to Claude for complex queries
- Configurable complexity threshold

### Adding New Extractors

See `src/extractors/base.py` for the `MetadataExtractor` interface.

Example:
```python
from src.extractors.base import MetadataExtractor

class MyLanguageExtractor(MetadataExtractor):
    def extract_metadata(self, source_path: Path) -> List[MetadataArtifact]:
        # Implementation
        pass
```

## API Endpoints

### `POST /api/query`
Semantic question-answering over metadata.

**Request:**
```json
{
  "query": "What validation constraints exist?",
  "filters": {
    "language": "java",
    "type": "field"
  },
  "top_k": 5
}
```

**Response:**
```json
{
  "answer": "...",
  "sources": [...],
  "retrieved_artifacts": [...],
  "metadata": {
    "query_time_ms": 234,
    "num_retrieved": 5
  }
}
```

### `POST /api/index/rebuild`
Trigger full metadata reindexing.

### `GET /api/index/status`
Get indexing progress and statistics.

### `GET /api/metadata`
Browse metadata directly with filters.

## Performance

### Local Stack (RTX 4090)
- Embedding: ~50ms per query
- Vector search: ~10ms
- LLM generation: ~2-5s (7B model, 4-bit)
- **Total latency**: ~2-6 seconds per query

### Claude Stack
- Embedding: ~50ms per query
- Vector search: ~10ms
- Claude API: ~1-3s
- **Total latency**: ~1-4 seconds per query

## Phase 1 Deliverables

- [x] Technology stack decisions
- [x] Project structure setup
- [ ] Metadata extractors (Java, TypeScript, Python)
- [ ] Unified indexing pipeline
- [ ] Embedding service
- [ ] ChromaDB integration
- [ ] Semantic search implementation
- [ ] Prompt engineering for grounded QA
- [ ] Answer generation with citations
- [ ] FastAPI endpoints
- [ ] Test query dataset
- [ ] Manual evaluation

## Next Phases

- **Phase 2**: Evaluation + Observability
- **Phase 3**: Agent Orchestration + Guardrails
- **Phase 4**: Frontend + Backend Integration

See [PROJECT_OVERVIEW_2026.md](../cheap-planning/PROJECT_OVERVIEW_2026.md) for full roadmap.

## License

Same as parent CHEAP project.
