# CHEAP RAG - Database & Code Metadata Semantic Search

Phase 1 implementation of the CHEAP AI Enhancement project: Core RAG pipeline with embeddings and vector search over **database schemas** and **multi-language code metadata**.

## Overview

This system provides semantic question-answering over database schemas (PostgreSQL, SQLite, MariaDB) and code metadata (Java, TypeScript, Python, Rust). It combines:

- **Database metadata extraction** from PostgreSQL, SQLite, MariaDB (tables, columns, relationships, indexes)
- **Code metadata extraction** from Java, TypeScript, Python, Rust source files
- **Unified metadata model** representing both databases and code
- **Semantic embeddings** using sentence-transformers
- **Vector search** with ChromaDB
- **LLM-powered generation** with citations using local models or Claude API

**Primary Use Case:** Enable semantic search over database schemas (e.g., Odoo ERP) with natural language queries like "What tables are related to sales orders?" or "How is the product table connected to inventory?"

**Secondary Use Case:** Unified search across database metadata AND code metadata (e.g., "Compare the Catalog interface in Java to the database table structure")

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

# Database credentials (for database metadata extraction)
ODOO_DB_PASSWORD=your_odoo_password_here
```

Edit `config/local.yaml` to configure database connections:

```yaml
indexing:
  # Database connections (for database metadata)
  databases:
    odoo_ecommerce:
      enabled: true  # Set to true when Odoo is installed
      type: "postgresql"
      connection:
        host: "localhost"
        port: 5432
        database: "odoo"
        user: "odoo"
        password: "${ODOO_DB_PASSWORD}"
      schema: "public"
      include_tables:  # Optional: limit to specific tables
        - "sale_order"
        - "product_product"
        # ... other tables
      tags: ["odoo", "ecommerce"]
```

### Usage

#### Demo: SQLite Database Extraction

Run the SQLite demo to see database metadata extraction in action:

```bash
# Create demo database and extract metadata
python scripts/demo_sqlite_extraction.py

# Review extracted metadata
cat data/metadata/demo_sqlite_metadata.json
```

This creates a simple eCommerce database (customers, products, orders) and extracts:
- Tables (4)
- Columns (20+)
- Relationships (foreign keys)
- Indexes

#### Demo: Java Code Extraction

Run the Java demo to see code metadata extraction:

```bash
# Extract metadata from CHEAP core Java interfaces
python scripts/demo_java_extraction.py

# Review extracted metadata
cat data/metadata/demo_java_metadata.json
```

This extracts metadata from CHEAP core Java source files:
- Interfaces (Catalog, Hierarchy, Entity, Aspect, Property)
- Classes and implementations
- Fields with types and constraints

#### 1. Index Database Metadata (When Odoo is installed)

```bash
# Extract from Odoo PostgreSQL database
python scripts/index_metadata.py --databases odoo_ecommerce

# Or extract from all configured databases
python scripts/index_metadata.py --databases all
```

#### 2. Index Code Metadata

```bash
# Extract from CHEAP Java source
python scripts/index_metadata.py --source ../cheap/cheap-core/src/main/java

# Or extract from multiple sources
python scripts/index_metadata.py \
  --source ../cheap/cheap-core/src/main/java \
  --source ../cheap-ts/src
```

#### 3. Query the System

```bash
# Interactive query
python scripts/query_example.py

# Example queries:
# - "What tables are related to sales orders?"
# - "Show me columns in the product_product table"
# - "How is sale_order connected to res_partner?"
# - "What is the Catalog interface in Java?"
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
