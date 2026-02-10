# CLAUDE.md - CHEAP RAG

This file provides guidance to Claude Code when working on the CHEAP RAG system.

## Project Overview

This is the Phase 1 implementation of the CHEAP AI Enhancement project: a RAG (Retrieval-Augmented Generation) system that enables semantic search over multi-language metadata definitions.

**Current Phase:** Phase 1 - Core RAG + Embeddings + Vector Search
**Timeline:** Weeks 1-2
**Status:** Initial setup complete, implementation in progress

## Technology Stack

See [../cheap-planning/TECH_STACK_DECISIONS.md](../cheap-planning/TECH_STACK_DECISIONS.md) for complete rationale.

**Core Stack:**
- Python 3.14
- Embeddings: sentence-transformers/all-mpnet-base-v2 (local GPU)
- Vector Store: ChromaDB (local persistence)
- LLM (Default): Qwen2.5-Coder-7B-Instruct via Ollama
- LLM (Alternate): Claude Sonnet 4.5 / Haiku 4.5 via API
- API: FastAPI

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

### Code Style

- **Python Version:** 3.14 (use modern type hints, pattern matching where appropriate)
- **Type Hints:** Required for all functions and methods
- **Formatting:** Black (line length 100)
- **Linting:** Ruff
- **Imports:** Absolute imports from `src.*`

### Configuration Management

- All configuration in YAML files under `config/`
- Three profiles: `local.yaml`, `claude.yaml`, `hybrid.yaml`
- Use `src/config.py` for loading and validation
- Never hardcode API keys or paths

### Testing

- Use pytest for all tests
- Aim for >80% coverage
- Mock external services (LLM APIs, file I/O) in unit tests
- Integration tests for end-to-end workflows
- Test file naming: `test_*.py`

### Metadata Extraction

Each language extractor must:
1. Inherit from `MetadataExtractor`
2. Return `List[MetadataArtifact]`
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

Follow the detailed plan in [../cheap-planning/PHASE_1_DETAILED_PLAN.md](../cheap-planning/PHASE_1_DETAILED_PLAN.md):

1. ✅ Technology stack decisions
2. ✅ Project structure setup
3. [ ] Metadata extraction (Java, TypeScript, Python)
4. [ ] Embedding service
5. [ ] Vector store integration
6. [ ] Semantic search
7. [ ] Answer generation with citations
8. [ ] API endpoints
9. [ ] Testing and evaluation

## Testing Strategy

### Unit Tests
- Each extractor independently
- Embedding service
- Vector store operations
- Prompt formatting

### Integration Tests
- Full RAG pipeline (extract → embed → index → search → generate)
- API endpoints
- Configuration loading

### Manual Evaluation
- Test query dataset (15-25 questions)
- Answer quality review
- Citation accuracy

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

## Key Decisions

1. **Local-first:** Default to local models to minimize cost and maximize privacy
2. **Configuration-driven:** Support multiple deployment modes without code changes
3. **Modular design:** Each component (extraction, embedding, retrieval, generation) is independent
4. **Citation-focused:** All answers must cite sources from retrieved context
5. **Quality over speed:** Prioritize answer quality in Phase 1; optimize in Phase 2

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
2. Inherit from `MetadataExtractor`
3. Implement `extract_metadata()` and `language()`
4. Add tests in `tests/test_extractors/test_{language}_extractor.py`
5. Update config `indexing.extractors` section

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
