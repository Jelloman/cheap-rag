#!/usr/bin/env python
"""Download and cache required models to the local models directory."""

from __future__ import annotations

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sentence_transformers import SentenceTransformer

MODELS_DIR = Path(__file__).parent.parent / "models" / "embeddings"

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"


def download_embedding_model() -> None:
    """Download the embedding model to the local models directory."""
    print(f"Downloading {EMBEDDING_MODEL} to {MODELS_DIR} ...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(EMBEDDING_MODEL, cache_folder=str(MODELS_DIR))

    dim = model.get_sentence_embedding_dimension()
    embedding = model.encode("Test embedding generation")
    print(f"  Embedding dimension : {dim}")
    print(f"  Test embedding shape: {embedding.shape}")
    print("✓ Embedding model ready")


def main() -> None:
    """Download all required models."""
    print("CHEAP RAG - Model Download")
    print("=" * 50)

    download_embedding_model()

    print("\n" + "=" * 50)
    print("All models downloaded successfully!")
    print("\nNext steps:")
    print("1. Install Ollama: https://ollama.ai")
    print("2. Pull Qwen model: ollama pull qwen2.5-coder:7b-instruct-q4_K_M")
    print("3. Run indexing: python scripts/index_metadata.py")


if __name__ == "__main__":
    main()
