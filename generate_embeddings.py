"""Generate embeddings and index Odoo metadata in ChromaDB."""

import json
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

import yaml
from dotenv import load_dotenv

from src.embeddings.service import EmbeddingService
from src.extractors.base import MetadataArtifact

from src.vectorstore.chroma_store import ChromaVectorStore

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Generate embeddings and index in vector store."""
    print("=" * 80)
    print("CHEAP RAG: Embedding Generation and Indexing (ChromaDB)")
    print("=" * 80)

    # Load configuration
    config_path = Path("config/local.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Load extracted metadata
    metadata_path = Path("data/metadata/odoo_ecommerce_metadata.json")
    if not metadata_path.exists():
        print(f"\n[ERROR] Metadata file not found: {metadata_path}")
        print("Please run extract_odoo_metadata.py first")
        return 1

    print(f"\n1. Loading metadata from: {metadata_path}")
    with open(metadata_path) as f:
        artifacts_data = json.load(f)

    # Convert to MetadataArtifact objects
    artifacts = [MetadataArtifact.from_dict(d) for d in artifacts_data]
    print(f"   [OK] Loaded {len(artifacts)} artifacts")

    # Group by type
    by_type = {}
    for artifact in artifacts:
        by_type[artifact.type] = by_type.get(artifact.type, 0) + 1

    print(f"   Breakdown:")
    for artifact_type, count in sorted(by_type.items()):
        print(f"     - {artifact_type}: {count}")

    # Initialize embedding service
    print(f"\n2. Initializing embedding service...")
    embedding_config = config["embedding"]

    # Use CPU if CUDA not available
    device = embedding_config.get("device", "cpu")
    if device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                print("   [WARNING] CUDA not available, falling back to CPU")
                device = "cpu"
        except ImportError:
            print("   [WARNING] PyTorch not installed, falling back to CPU")
            device = "cpu"

    embedding_service = EmbeddingService(
        model_name=embedding_config["model_name"],
        device=device,
        cache_dir=embedding_config.get("cache_dir"),
        batch_size=embedding_config.get("batch_size", 32),
    )
    print(f"   [OK] Model loaded: {embedding_config['model_name']}")
    print(f"   Embedding dimension: {embedding_service.get_dimension()}")
    print(f"   Device: {device}")

    # Generate embeddings
    print(f"\n3. Generating embeddings...")
    print(f"   Processing {len(artifacts)} artifacts...")

    # Show a sample embedding text
    if artifacts:
        print(f"\n   Sample embedding text (first artifact):")
        print(f"   {'-' * 70}")
        sample_text = artifacts[0].to_embedding_text()
        # Truncate if too long
        if len(sample_text) > 200:
            print(f"   {sample_text[:200]}...")
        else:
            print(f"   {sample_text}")
        print(f"   {'-' * 70}")

    embeddings = embedding_service.embed_artifacts(artifacts)
    print(f"   [OK] Generated {len(embeddings)} embeddings")
    print(f"   Embedding shape: {embeddings.shape}")

    # Initialize vector store
    print(f"\n4. Initializing FAISS vector store...")
    vectorstore_config = config["vectorstore"]

    vector_store = ChromaVectorStore(
        persist_directory=vectorstore_config["persist_directory"],
        collection_name=vectorstore_config["collection_name"],
        distance_metric=vectorstore_config["distance_metric"],
    )

    current_count = vector_store.count()
    print(f"   [OK] Vector store initialized")
    print(f"   Current items in collection: {current_count}")

    # Ask user if they want to clear existing data
    if current_count > 0:
        print(f"\n   [WARNING] Collection already contains {current_count} items")
        response = input("   Clear existing data? (y/N): ").strip().lower()
        if response == "y":
            vector_store.delete_all()
            print("   [OK] Existing data cleared")
        else:
            print("   [OK] Keeping existing data (will add new items)")

    # Add artifacts to vector store
    print(f"\n5. Indexing artifacts in ChromaDB vector store...")
    vector_store.add_artifacts(artifacts, embeddings)
    print(f"   [OK] Indexing complete!")
    print(f"   Total items in collection: {vector_store.count()}")

    # Verify indexing with a test query
    print(f"\n6. Verification: Testing semantic search...")
    test_queries = [
        "What tables are related to sales orders?",
        "Show me product information",
        "How are customers stored?",
    ]

    for query in test_queries:
        print(f"\n   Query: \"{query}\"")
        query_embedding = embedding_service.embed_text(query)
        ids, metadatas, distances = vector_store.search(
            query_embedding, top_k=3
        )

        if ids:
            print(f"   Top 3 results:")
            for i, (artifact_id, metadata, distance) in enumerate(
                zip(ids, metadatas, distances), 1
            ):
                similarity = 1 - distance  # Convert distance to similarity
                print(f"     {i}. {metadata.get('name', 'Unknown')} "
                     f"({metadata.get('type', 'unknown')}) "
                     f"- Similarity: {similarity:.3f}")
        else:
            print("   No results found")

    # Summary
    print("\n" + "=" * 80)
    print("Indexing Complete!")
    print("=" * 80)
    print(f"\nSummary:")
    print(f"  - Artifacts indexed: {len(artifacts)}")
    print(f"  - Embedding model: {embedding_config['model_name']}")
    print(f"  - Embedding dimension: {embedding_service.get_dimension()}")
    print(f"  - Vector store: ChromaDB ({vectorstore_config['collection_name']})")
    print(f"  - Distance metric: {vectorstore_config['distance_metric']}")
    print(f"  - Total items in collection: {vector_store.count()}")

    print("\n" + "=" * 80)
    print("Next Steps:")
    print("=" * 80)
    print("1. Test semantic search with more queries")
    print("2. Implement retrieval service (Week 2, Days 3-4)")
    print("3. Integrate LLM for answer generation")
    print("4. Create API endpoints")

    return 0


if __name__ == "__main__":
    sys.exit(main())
