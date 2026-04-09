# CHEAP RAG Setup Guide

Quick start guide for setting up the CHEAP RAG development environment.

## Prerequisites

- **Python 3.13** installed (ChromaDB is incompatible with 3.14)
- **UV** package manager installed (`pip install uv` or see [uv docs](https://docs.astral.sh/uv/))
- **CUDA-capable GPU** with 8GB+ VRAM (recommended: RTX 4090 24GB)
- **Git** for version control
- **Ollama** for local LLM inference (optional but recommended)

## Setup Steps

### 1. Install Dependencies

```bash
cd cheap-rag

# Install all dependencies (including dev extras)
uv sync --extra dev
```

This installs Python 3.13 automatically if needed, creates a virtual environment, and installs all dependencies including CUDA-enabled PyTorch.

### 2. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env and set:
# - CONFIG_PROFILE=local (or claude, or hybrid)
# - ANTHROPIC_API_KEY=your_key_here (if using Claude mode)
```

### 3. Download Embedding Model

```bash
# This will download sentence-transformers/all-mpnet-base-v2
uv run python scripts/download_models.py
```

Expected output:
```
Downloading sentence-transformers/all-mpnet-base-v2...
Model downloaded successfully
Embedding dimension: 768
✓ Embedding model ready
```

### 4. Install Ollama (for local LLM)

**Skip this step if using Claude API only.**

```bash
# Download from https://ollama.ai
# Or install via package manager:

# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows: Download installer from https://ollama.ai
```

Pull the Qwen model:
```bash
ollama pull qwen2.5-coder:7b-instruct-q4_K_M
```

Verify Ollama is running:
```bash
ollama list
# Should show qwen2.5-coder:7b-instruct-q4_K_M
```

### 5. Verify Installation

```bash
# Test configuration loading
uv run python -c "from src.config import load_config; print(load_config())"

# Should print configuration without errors
```

### 6. Create Log Directory

```bash
mkdir -p logs
```

## Configuration Profiles

### Local Mode (Default)

All processing on local GPU. No API costs.

```bash
# .env
CONFIG_PROFILE=local
```

**Requirements:**
- CUDA GPU with 8GB+ VRAM
- Ollama running with qwen2.5-coder model

### Claude Mode

Uses Claude API for generation. Local embeddings.

```bash
# .env
CONFIG_PROFILE=claude
ANTHROPIC_API_KEY=sk-ant-...
```

**Requirements:**
- Anthropic API key
- API costs apply (~$0.003-0.015 per query)

### Hybrid Mode

Automatic routing between local and Claude.

```bash
# .env
CONFIG_PROFILE=hybrid
ANTHROPIC_API_KEY=sk-ant-...
```

**Requirements:**
- Both Ollama and Claude API set up
- Uses local by default, Claude for complex queries

## Testing Installation

### Test Embedding Model

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
embedding = model.encode("Test text")
print(f"Embedding shape: {embedding.shape}")  # Should be (768,)
```

### Test Ollama Connection

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "qwen2.5-coder:7b-instruct-q4_K_M",
  "prompt": "Say hello",
  "stream": false
}'
```

### Test Claude API (if using)

```python
import os
from anthropic import Anthropic

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
message = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=100,
    messages=[{"role": "user", "content": "Say hello"}]
)
print(message.content)
```

## Next Steps

After setup is complete:

1. **Index Metadata**
   ```bash
   uv run python scripts/index_metadata.py --source ../cheap-core/src/main/java
   ```

2. **Run Test Queries**
   ```bash
   uv run python scripts/query_example.py
   ```

3. **Start API Server**
   ```bash
   uv run uvicorn src.api.routes:app --reload
   ```

## Troubleshooting

### CUDA Not Available

If you see "CUDA not available":
- Verify GPU drivers installed
- Check PyTorch CUDA version: `uv run python -c "import torch; print(torch.cuda.is_available())"`
- Reinstall PyTorch with CUDA: Visit https://pytorch.org/get-started/locally/

### Ollama Connection Failed

If Ollama connection fails:
- Check Ollama is running: `ollama list`
- Verify port 11434 is accessible: `curl http://localhost:11434/api/tags`
- Restart Ollama: `ollama serve`

### ChromaDB Errors

If ChromaDB fails to initialize:
- Delete `data/vector_db/` and recreate
- Check disk space
- Verify write permissions

### Import Errors

If module imports fail:
- Verify all dependencies installed: `uv sync --extra dev`
- Try reinstalling: `uv sync --extra dev --reinstall`

## Development Workflow

1. Set CONFIG_PROFILE in .env
2. Run tests: `nox -s tests`
3. Make changes
4. Run tests again: `nox -s tests`
5. Commit changes

## Hardware Requirements

### Minimum (CPU-only mode)
- CPU: 4+ cores
- RAM: 16GB
- Disk: 20GB

### Recommended (GPU mode)
- GPU: 8GB+ VRAM (RTX 3060 or better)
- CPU: 8+ cores
- RAM: 32GB
- Disk: 50GB (for models and data)

### Optimal (Development)
- GPU: 24GB VRAM (RTX 4090)
- CPU: 12+ cores
- RAM: 64GB
- Disk: 100GB SSD

## Cost Estimates

### Local Stack
- **Infrastructure:** $0 (uses existing hardware)
- **API calls:** $0
- **Electricity:** ~$0.10-0.50/day (GPU power)

### Claude Stack
- **Embeddings:** $0 (local)
- **Generation:** ~$0.003-0.015/query
- **Estimated:** $3-15/day for active development

### Hybrid Stack
- **Most queries:** $0 (local)
- **Complex queries:** ~$0.003-0.015/query
- **Estimated:** $1-5/day

## Useful Commands

```bash
# Run all checks (tests + typecheck + lint)
nox

# Run tests with coverage
nox -s tests

# Format code
nox -s format

# Lint code
nox -s lint

# Type check
nox -s typecheck

# Start development server
uv run uvicorn src.api.routes:app --reload --port 8000

# Index metadata
uv run python scripts/index_metadata.py

# Query via CLI
uv run python scripts/query_example.py

# Check Ollama models
ollama list

# Pull latest model
ollama pull qwen2.5-coder:7b-instruct-q4_K_M

# View logs
tail -f logs/cheap-rag.log
```

## Project Status

**Phase 1 — Core RAG ✅ Complete**
- [x] Metadata extractors (Java, Python, TypeScript, PostgreSQL, SQLite)
- [x] Embedding service (sentence-transformers, GPU-accelerated)
- [x] Vector store (ChromaDB)
- [x] Semantic search with metadata filtering
- [x] LLM answer generation with citations (Ollama + Claude)
- [x] API endpoints

**Phase 2 — Evaluation + Observability ✅ Complete**
- [x] Retrieval metrics (Precision@K, Recall@K, MRR, MAP, NDCG)
- [x] Gold dataset system
- [x] OpenTelemetry tracing infrastructure
- [x] Structured logging, error tracking, performance profiling
- [x] A/B testing framework

**Phase 2.5 — Observability Integration ✅ Complete**
- [x] Tracing wired throughout pipeline
- [x] Per-request correlation IDs
- [x] 324 tests passing

**Phase 3 — Runtime Observability & Metrics ⏳ In Progress**

See [README.md](../README.md) for full project overview.
