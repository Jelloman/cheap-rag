#!/usr/bin/env python
"""Download and cache required models."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sentence_transformers import SentenceTransformer


def download_embedding_model():
    """Download the embedding model."""
    print("Downloading sentence-transformers/all-mpnet-base-v2...")
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    print(f"Model downloaded successfully: {model}")
    print(f"Embedding dimension: {model.get_sentence_embedding_dimension()}")

    # Test encoding
    test_text = "Test embedding generation"
    embedding = model.encode(test_text)
    print(f"Test embedding shape: {embedding.shape}")
    print("✓ Embedding model ready")


def main():
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
