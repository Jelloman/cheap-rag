# CHEAP RAG Setup Guide

Quick start guide for setting up the CHEAP RAG development environment.

## Prerequisites

- **Python 3.14** installed
- **CUDA-capable GPU** with 8GB+ VRAM (recommended: RTX 4090 24GB)
- **Git** for version control
- **Ollama** for local LLM inference (optional but recommended)

## Setup Steps

### 1. Create Virtual Environment

```bash
cd cheap-rag

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate
```

### 2. Install Dependencies

```bash
# Install all dependencies
pip install -r requirements.txt

# Or with development dependencies
pip install -e ".[dev]"
```

### 3. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env and set:
# - CONFIG_PROFILE=local (or claude, or hybrid)
# - ANTHROPIC_API_KEY=your_key_here (if using Claude mode)
```

### 4. Download Embedding Model

```bash
# This will download sentence-transformers/all-mpnet-base-v2
python scripts/download_models.py
```

Expected output:
```
Downloading sentence-transformers/all-mpnet-base-v2...
Model downloaded successfully
Embedding dimension: 768
✓ Embedding model ready
```

### 5. Install Ollama (for local LLM)

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

### 6. Verify Installation

```bash
# Test configuration loading
python -c "from src.config import load_config; print(load_config())"

# Should print configuration without errors
```

### 7. Create Log Directory

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

1. **Index Metadata** (not yet implemented)
   ```bash
   python scripts/index_metadata.py --source ../cheap-core/src/main/java
   ```

2. **Run Test Queries** (not yet implemented)
   ```bash
   python scripts/query_example.py
   ```

3. **Start API Server** (not yet implemented)
   ```bash
   uvicorn src.api.routes:app --reload
   ```

## Troubleshooting

### CUDA Not Available

If you see "CUDA not available":
- Verify GPU drivers installed
- Check PyTorch CUDA version: `python -c "import torch; print(torch.cuda.is_available())"`
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
- Ensure virtual environment is activated
- Verify all dependencies installed: `pip list`
- Try reinstalling: `pip install -r requirements.txt --force-reinstall`

## Development Workflow

1. Activate virtual environment
2. Set CONFIG_PROFILE in .env
3. Run tests: `pytest`
4. Make changes
5. Run tests again
6. Commit changes

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
# Run tests
pytest

# Run tests with coverage
pytest --cov=src --cov-report=html

# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Type check
mypy src/

# Start development server
uvicorn src.api.routes:app --reload --port 8000

# Check Ollama models
ollama list

# Pull latest model
ollama pull qwen2.5-coder:7b-instruct-q4_K_M

# View logs
tail -f logs/cheap-rag.log
```

## Project Status

**Phase 1 - Setup Complete ✓**
- [x] Technology decisions finalized
- [x] Project structure created
- [x] Configuration system implemented
- [x] Base classes defined

**Phase 1 - Implementation In Progress**
- [ ] Metadata extractors (Java, TypeScript, Python)
- [ ] Embedding service
- [ ] Vector store integration
- [ ] Semantic search
- [ ] Answer generation
- [ ] API endpoints
- [ ] Testing and evaluation

See [README.md](README.md) for full project overview.
