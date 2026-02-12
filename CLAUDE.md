# CLAUDE.md - CHEAP RAG

This file provides guidance to Claude Code when working on the CHEAP RAG system.

## Project Overview

This is the Phase 1 implementation of the CHEAP AI Enhancement project: a RAG (Retrieval-Augmented Generation) system that enables semantic search over multi-language metadata definitions.

**Current Phase:** Phase 1 - Core RAG + Embeddings + Vector Search
**Status:** Core implementation complete, integration testing in progress

**Recent Progress (2026-02-11):**
- ✅ Retrieval layer implemented and tested (85% coverage on semantic_search.py)
- ✅ Generation layer implemented (Ollama + Claude providers, citation extraction)
- ✅ API endpoints implemented (FastAPI with CORS, query/index/metadata routes)
- ✅ Indexing pipeline implemented (unified extractor registration, error handling)
- ✅ Development tooling complete (nox, BasedPyright strict mode, PyTorch CUDA 12.4)
- ✅ 15 tests passing with 21% overall coverage (retrieval tested, generation/API need tests)

**Next Steps:**
- Integration testing for generation and API layers
- Manual evaluation of answer quality
- Performance benchmarking

See [../cheap-planning/PROJECT_STATUS.md](../cheap-planning/PROJECT_STATUS.md) for overall project status.
See [../cheap-planning/TODO_PHASE_1.md](../cheap-planning/TODO_PHASE_1.md) for remaining tasks.

## Technology Stack

See [../cheap-planning/attic/TECH_STACK_DECISIONS.md](../cheap-planning/attic/TECH_STACK_DECISIONS.md) for complete rationale.

**Core Stack:**
- Python 3.13 (ChromaDB incompatible with 3.14)
- PyTorch: 2.6.0+cu124 (CUDA 12.4 for GPU acceleration)
- Embeddings: sentence-transformers/all-mpnet-base-v2 (local GPU)
- Vector Store: ChromaDB (local persistence)
- LLM (Default): Qwen2.5-Coder-7B-Instruct via Ollama
- LLM (Alternate): Claude Sonnet 4.5 / Haiku 4.5 via API
- API: FastAPI
- Dev Tools: BasedPyright (strict), Ruff (lint/format), Nox (tasks), pytest

## Project Structure

```
cheap-rag/
├── config/               # YAML configuration files
├── src/
│   ├── extractors/       # Language-specific metadata extraction
│   ├── indexing/         # Pipeline and schema
│   ├── embeddings/       # Embedding service
│   ├── vectorstore/      # ChromaDB integration
│   ├── retrieval/        # Search and filtering
│   ├── generation/       # LLM answer generation
│   └── api/              # FastAPI endpoints
├── tests/                # Test suites
├── data/                 # Runtime data (gitignored)
├── scripts/              # Utility scripts
└── requirements.txt
```

## Development Guidelines

### Code Style & Tooling

- **Python Version:** 3.13 (ChromaDB incompatible with 3.14)
- **Build System:** Hatchling + UV workspace
- **Type Hints:**
  - Required for all functions and methods (100% coverage target)
  - Use PEP 604 syntax: `T | None`, `list[T]`, `dict[K, V]`
  - Add `from __future__ import annotations` to all files
- **Type Checking:** BasedPyright in strict mode (see `pyrightconfig.json`)
- **Formatting:** Ruff (line length 100)
- **Linting:** Ruff (comprehensive rule set)
- **Testing:** pytest with nox automation
- **Imports:** Absolute imports from `src.*`
- **Interfaces:** Use `@runtime_checkable Protocol` for structural typing
- **PEP 561 Compliance:** Package includes `py.typed` marker for type distribution

### Configuration Management

**Application Configuration:**
- All configuration in YAML files under `config/`
- Three profiles: `local.yaml`, `claude.yaml`, `hybrid.yaml`
- Use `src/config.py` for loading and validation
- Never hardcode API keys or paths

**Development Tool Configuration:**
- `pyproject.toml` - Build system, dependencies, tool configs (Ruff, pytest, coverage)
- `pyrightconfig.json` - BasedPyright strict mode settings
- `noxfile.py` - Task automation (tests, typecheck, lint, format)
- `src/py.typed` - PEP 561 compliance marker for type distribution

### Development Tools

This project uses a modern Python development stack with strict type safety and automated quality checks.

**Quick Commands (via Nox):**
```bash
# Run all quality checks (tests, typecheck, lint)
nox

# Individual sessions
nox -s tests          # Run tests with coverage
nox -s typecheck      # Type check with BasedPyright
nox -s lint           # Lint and format check with Ruff
nox -s format         # Auto-format code with Ruff

# Pass arguments to underlying tools
nox -s tests -- -v -k test_specific
nox -s typecheck -- src/specific_file.py
```

**Hatch Scripts (alternative):**
```bash
# Via hatch (if you prefer hatch over nox)
hatch run test
hatch run typecheck
hatch run lint
hatch run format
```

**Direct Tool Usage:**
```bash
# Type checking (strict mode)
basedpyright src/

# Linting
ruff check src/

# Formatting
ruff format src/

# Testing
pytest tests/ -v --cov=src
```

### Testing

- **Framework:** pytest with coverage reporting
- **Automation:** nox for isolated test environments
- **Coverage Target:** >80% code coverage
- **Test Types:**
  - Unit tests: Each extractor independently
  - Integration tests: Full RAG pipeline
  - Mock external services (LLM APIs, file I/O)
- **Test file naming:** `test_*.py`
- **Run tests:** `nox -s tests` or `pytest tests/`

### Type Safety Requirements

**All code must:**
1. Pass BasedPyright strict mode with zero errors
2. Have 100% type hint coverage
3. Use modern type syntax (PEP 604)
4. Include `from __future__ import annotations`

**Type stubs:** Automatically resolved by BasedPyright via typeshed - no manual stub packages needed.

**Before committing:**
```bash
# Ensure all checks pass
nox
```

### Metadata Extraction

Each language extractor must:
1. Implement the `MetadataExtractor` Protocol (structural typing)
2. Return `list[MetadataArtifact]` (note: lowercase `list`)
3. Generate unique IDs (hash of qualified name + language)
4. Extract descriptions from docs/comments
5. Handle errors gracefully (log and continue)

**Artifact Fields:**
- Required: id, name, type, language, module, description
- Optional: constraints, relations, examples, tags, metadata

**Embedding Text Format:**
Use `MetadataArtifact.to_embedding_text()` for consistency.

### Vector Store

- Use ChromaDB client from `src/vectorstore/`
- Store full MetadataArtifact as metadata
- Enable filtering on: language, type, module, tags
- Cosine similarity for distance metric

### LLM Integration

**Prompt Engineering:**
- System message emphasizes "use only provided context"
- Require explicit citations
- Format: `[ArtifactName] (ID: artifact_id)`
- Handle "I don't know" cases gracefully

**Provider Abstraction:**
- Support both Ollama and Anthropic via same interface
- Configuration-driven selection
- Hybrid mode routes based on complexity

### API Design

- Use FastAPI with Pydantic models
- Async handlers where applicable
- Error responses with detail
- CORS configured for local development

## Phase 1 Implementation Order

See [../cheap-planning/TODO_PHASE_1.md](../cheap-planning/TODO_PHASE_1.md) for detailed task status.

1. ✅ Technology stack decisions
2. ✅ Project structure setup
3. ✅ Metadata extraction (Java, PostgreSQL, SQLite)
4. ✅ Embedding service
5. ✅ Vector store integration
6. ✅ Semantic search (implemented and tested, 85% coverage)
7. ✅ Answer generation with citations (implemented, needs integration tests)
8. ✅ API endpoints (implemented, needs integration tests)
9. ⏳ Testing and evaluation (retrieval tested, integration tests in progress)

## Testing Strategy

### Unit Tests
- ✅ MetadataArtifact model (10 tests passing)
- Each extractor independently (basic tests exist)
- Embedding service (partial coverage)
- Vector store operations (partial coverage)
- Prompt formatting (needs tests)

### Integration Tests
- ⏳ Full RAG pipeline (extract → embed → index → search → generate)
- API endpoints (needs tests)
- Configuration loading (90% coverage)
- ✅ Retrieval quality tests (Precision@K, Recall@K, MRR - 5 tests passing)

### Manual Evaluation
- ✅ Test query dataset created (20+ questions in tests/fixtures/test_queries.json)
- ⏳ Answer quality review (needs testing)
- ⏳ Citation accuracy (needs testing)

**Current Test Status (2026-02-11):**
- 15 tests passing
- 21% overall code coverage
- 85% coverage on semantic_search.py
- 67% coverage on filters.py
- 0% coverage on generation/* and api/* (implemented but not tested)

## Common Patterns

### Loading Configuration
```python
from src.config import load_config

config = load_config()  # Uses CONFIG_PROFILE env var
# or
config = load_config("config/local.yaml")
```

### Creating Metadata Artifacts
```python
from src.extractors.base import MetadataArtifact

artifact = MetadataArtifact(
    id="java_Catalog_abc123",
    name="Catalog",
    type="interface",
    language="java",
    module="com.example.cheap.core",
    description="Root container for CHEAP data",
    tags=["core", "interface"],
    source_file="Catalog.java",
    source_line=10,
)
```

### Embedding Generation
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
embedding = model.encode(artifact.to_embedding_text())
```

## Development Tooling Decisions

### Why These Tools?

**BasedPyright over mypy:**
- Faster type checking (written in Node.js vs Python)
- Better type inference and error messages
- Automatic type stub resolution via typeshed
- Stricter mode catches more issues
- Community-driven fork of Pyright with active development

**Ruff over Black + Flake8:**
- 10-100x faster (written in Rust)
- Handles both formatting and linting in one tool
- Drop-in replacement for Black, Flake8, isort, and more
- Comprehensive rule set with auto-fix capabilities
- Single tool reduces complexity

**Nox over tox:**
- Simpler Python-based configuration (vs INI)
- Better integration with modern tools (UV, Hatch)
- Explicit session definitions
- Easier to customize and extend

**UV over pip:**
- Significantly faster dependency resolution
- Built-in workspace support
- Better lockfile handling
- Modern Python packaging tool
- Recommended for new projects

**Hatchling over setuptools:**
- Modern build backend (PEP 517/518)
- Simpler configuration
- Better defaults
- Active development

**Protocol over ABC:**
- Structural typing (duck typing)
- No inheritance required
- More flexible for testing (easier mocking)
- Clearer interface definitions
- Supports runtime type checking with `@runtime_checkable`

### Type Safety Philosophy

This project enforces strict type safety:
- **100% type hint coverage** - Every function has complete type annotations
- **Strict mode** - BasedPyright's strictest settings enabled
- **PEP 604 syntax** - Modern type hints (`T | None`, not `Optional[T]`)
- **No `Any` escapes** - Minimize use of `Any` type
- **Protocol-based interfaces** - Use Protocols for structural typing

Type safety catches bugs at development time, improves IDE support, and serves as living documentation.

## Key Decisions

1. **Local-first:** Default to local models to minimize cost and maximize privacy
2. **Configuration-driven:** Support multiple deployment modes without code changes
3. **Modular design:** Each component (extraction, embedding, retrieval, generation) is independent
4. **Citation-focused:** All answers must cite sources from retrieved context
5. **Quality over speed:** Prioritize answer quality in Phase 1; optimize in Phase 2
6. **Type safety:** Strict type checking prevents bugs and improves maintainability

## External Dependencies

### Ollama Setup
```bash
# Install from https://ollama.ai
ollama pull qwen2.5-coder:7b-instruct-q4_K_M
ollama serve
```

### Claude API
- Requires `ANTHROPIC_API_KEY` environment variable
- Used in `claude` and `hybrid` modes
- Track costs via `cost_tracking` config

## Performance Targets (Phase 1)

- **Embedding:** ~50ms per query
- **Vector search:** ~10-50ms
- **LLM generation:** ~2-5s (local), ~1-3s (Claude)
- **Total latency:** <10s per query

Optimization is Phase 2 work; focus on correctness in Phase 1.

## Common Tasks

### Adding a New Language Extractor
1. Create `src/extractors/{language}_extractor.py`
2. Implement `MetadataExtractor` Protocol (no explicit inheritance needed)
3. Add `from __future__ import annotations` at top
4. Implement `extract_metadata()` and `language()` methods
5. Add type hints for all parameters and returns
6. Add tests in `tests/test_extractors/test_{language}_extractor.py`
7. Run `nox` to ensure all checks pass
8. Update config `indexing.extractors` section

### Adding New Configuration Profile
1. Copy existing `config/*.yaml`
2. Modify relevant sections
3. Update `.env.example` if new env vars needed
4. Document in README.md

### Running the Full Pipeline
```bash
# 1. Index metadata
python scripts/index_metadata.py

# 2. Query
python scripts/query_example.py

# 3. Or via API
uvicorn src.api.routes:app --reload
```

## Error Handling

- Log errors with context (file, line, artifact)
- Continue processing on non-fatal errors
- Collect and report errors at end of batch
- Use `try/except` in extractors to handle malformed code

## Documentation

- Docstrings required for all public functions/classes
- Use Google-style docstrings
- Include examples in docstrings for complex functions
- Update README.md when adding major features

## Git Workflow

- Commit extracted metadata JSON separately from code
- Don't commit downloaded models (in .gitignore)
- Don't commit .env files
- Commit vector DB schema changes with code

## Phase 1 Success Criteria

- [ ] Can ingest metadata from Java and TypeScript (minimum)
- [ ] Semantic search returns relevant artifacts for test queries
- [ ] Generated answers include proper citations
- [ ] Metadata filters work correctly (language, type, tags)
- [ ] API endpoints are functional
- [ ] Retrieval Precision@5 > 0.6
- [ ] Answer relevance rate > 0.7 on manual evaluation
- [ ] Average query latency < 10 seconds

## Next Phase Preview

Phase 2 will add:
- Evaluation framework (Precision@K, Recall@K, MRR)
- Observability (OpenTelemetry, tracing)
- Performance profiling and optimization
- A/B testing for embedding models

Don't optimize prematurely; focus on Phase 1 deliverables first.
