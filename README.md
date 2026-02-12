# CHEAP RAG - Semantic Search Over Metadata

Phase 1 implementation of the CHEAP AI Enhancement project: a RAG (Retrieval-Augmented Generation) system that enables semantic search and natural language Q&A over multi-language metadata definitions.

## Quick Start

### 1. Installation

```bash
cd cheap-rag

# Install dependencies with development extras using UV
uv sync --extra dev

# Or with pip
pip install -e ".[dev]"
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

## Development

This project uses modern Python tooling with strict type safety and automated quality checks.

### Setup Development Environment

```bash
# Install with development dependencies
uv sync --extra dev

# Verify installation
nox --list
```

### Development Workflow

**Run all quality checks:**
```bash
nox                    # Runs tests, type checking, and linting
```

**Individual checks:**
```bash
nox -s tests           # Run pytest with coverage
nox -s typecheck       # Type check with BasedPyright (strict mode)
nox -s lint            # Lint and format check with Ruff
nox -s format          # Auto-format code with Ruff
```

**Pass arguments to underlying tools:**
```bash
nox -s tests -- -v -k test_query        # Run specific tests
nox -s typecheck -- src/api/routes.py   # Type check specific file
nox -s lint -- --fix                    # Auto-fix lint issues
```

### Type Safety

This project enforces **100% type hint coverage** with BasedPyright in strict mode:

```bash
# Type check entire codebase
nox -s typecheck

# Type check with statistics
basedpyright --stats src/
```

**Type system requirements:**
- All functions and methods must have type hints
- Use PEP 604 syntax: `T | None`, `list[T]`, `dict[K, V]`
- Include `from __future__ import annotations` in all files
- Define interfaces using `@runtime_checkable Protocol`

### Code Quality

**Linting and formatting:**
```bash
# Check code style
nox -s lint

# Auto-format code
nox -s format

# Or use ruff directly
ruff check src/        # Check for issues
ruff format src/       # Format code
```

**Ruff configuration:**
- Line length: 100 characters
- Enabled rules: pycodestyle, pyflakes, bugbear, comprehensions, pyupgrade, unused-arguments, simplify
- See `pyproject.toml` for complete configuration

### Testing

```bash
# Run all tests with coverage
nox -s tests

# Run specific test file
pytest tests/test_extractors/test_java_extractor.py -v

# Run with coverage report
pytest --cov=src --cov-report=html tests/

# View coverage report
open htmlcov/index.html
```

**Testing standards:**
- Minimum 80% code coverage
- Unit tests for all extractors, services, and utilities
- Integration tests for end-to-end workflows
- Mock external services (LLM APIs, databases)

### Hatch Commands (Alternative)

If you prefer Hatch over Nox:

```bash
hatch run test         # Run tests
hatch run typecheck    # Type checking
hatch run lint         # Linting
hatch run format       # Formatting
```

### Build System

This project uses:
- **Build backend:** Hatchling
- **Package manager:** UV (recommended) or pip
- **Task runner:** Nox (primary) or Hatch (alternative)
- **Type checker:** BasedPyright (strict mode)
- **Linter/Formatter:** Ruff

### Pre-commit Checklist

Before committing code:

1. **Format code:** `nox -s format`
2. **Run all checks:** `nox` (tests + typecheck + lint)
3. **Verify all pass:** Zero errors in all sessions

### CI/CD

The project is configured for automated checks:
- Type checking: BasedPyright strict mode must pass
- Linting: Ruff checks must pass
- Testing: All tests must pass with >80% coverage
- Formatting: Code must be formatted with Ruff

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

### Troubleshooting

**Type checking errors:**
```bash
# Get detailed error information
basedpyright --verbose src/

# Check type coverage statistics
basedpyright --stats src/
```

**Linting issues:**
```bash
# Auto-fix most issues
ruff check src/ --fix

# Show detailed explanations
ruff check src/ --output-format=full
```

**Test failures:**
```bash
# Run with verbose output
pytest -vv tests/

# Run with debugging
pytest --pdb tests/

# Run specific test
pytest tests/test_file.py::test_function -v
```

**Import errors:**
```bash
# Reinstall in editable mode
uv sync --extra dev

# Verify installation
python -c "import src; print(src.__file__)"
```

**Nox issues:**
```bash
# Clear nox cache and rebuild
nox --clean

# List available sessions
nox --list

# Run with verbose output
nox -s tests -- -v
```

## Development Status

All Phase 1 core components complete:
- ✅ Retrieval Layer (semantic search, filters)
- ✅ LLM Answer Generation (Ollama + Claude providers)
- ✅ Citation extraction and validation
- ✅ API endpoints
- ✅ Indexing pipeline
- ✅ Test query dataset

See `../cheap-planning/TODO_PHASE_1.md` for detailed status.

## Quick Reference

### Essential Commands

```bash
# Setup
uv sync --extra dev              # Install dependencies

# Development
nox                              # Run all checks
nox -s format                    # Format code
nox -s tests                     # Run tests
nox -s typecheck                 # Type check
nox -s lint                      # Lint code

# Running
python scripts/index_metadata.py # Index metadata
python scripts/query_example.py  # Query via CLI
uvicorn src.api.routes:app       # Start API server

# Debugging
pytest -vv --pdb                 # Debug tests
basedpyright --verbose src/      # Detailed type errors
ruff check src/ --output-format=full  # Detailed lint info
```

### File Structure

```
cheap-rag/
├── src/                    # Source code
│   ├── extractors/         # Metadata extraction
│   ├── embeddings/         # Embedding generation
│   ├── vectorstore/        # Vector storage (ChromaDB, FAISS)
│   ├── retrieval/          # Semantic search
│   ├── generation/         # LLM answer generation
│   ├── api/                # FastAPI endpoints
│   ├── indexing/           # Pipeline and schema
│   └── config.py           # Configuration loading
├── tests/                  # Test suite
├── config/                 # YAML configuration profiles
├── scripts/                # Utility scripts
├── noxfile.py             # Task automation
├── pyrightconfig.json     # Type checker config
└── pyproject.toml         # Build and tool config
```

### Configuration Files

- **pyproject.toml** - Dependencies, build config, tool settings
- **pyrightconfig.json** - BasedPyright strict type checking
- **noxfile.py** - Automated test/lint/format tasks
- **src/py.typed** - PEP 561 type distribution marker
- **config/*.yaml** - Application runtime configuration

## License

Part of the CHEAP metadata framework project.
