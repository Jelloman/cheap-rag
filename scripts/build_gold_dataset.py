"""Build gold evaluation dataset from test queries.

This script:
1. Loads test queries from tests/fixtures/test_queries.json
2. Runs each query through the vector store
3. Retrieves top-K candidates
4. Saves results for manual annotation

The output requires manual review to select truly relevant artifacts.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.evaluation.gold_dataset import build_gold_dataset_from_index
from src.vectorstore.chroma_store import ChromaStore


def main() -> None:
    """Build gold dataset from test queries."""
    # Load configuration
    config = load_config()

    # Initialize vector store
    vector_store = ChromaStore(config)

    # Paths
    test_queries_path = Path(__file__).parent.parent / "tests" / "fixtures" / "test_queries.json"
    output_path = (
        Path(__file__).parent.parent / "tests" / "fixtures" / "gold_dataset.json"
    )

    print("Building gold dataset from test queries...")
    print(f"Input: {test_queries_path}")
    print(f"Output: {output_path}")
    print()

    # Build gold dataset
    dataset = build_gold_dataset_from_index(
        vector_store=vector_store,
        test_queries_path=test_queries_path,
        output_path=output_path,
        top_k=10,
    )

    print(f"Generated {len(dataset)} gold queries")
    print()
    print("IMPORTANT: This dataset requires manual review!")
    print("Review the output file and:")
    print("1. Verify that relevant_artifact_ids contains only truly relevant artifacts")
    print("2. Remove any false positives")
    print("3. Add any missing relevant artifacts")
    print("4. Update the 'requires_manual_review' metadata field to False")
    print()
    print(f"Dataset saved to: {output_path}")


if __name__ == "__main__":
    main()
