#!/usr/bin/env python3
"""Demonstration script for Python and TypeScript extractors.

This script shows how to use the new extractors to process code from the
cheap-py and cheap-ts projects.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extractors import PythonExtractor, TypeScriptExtractor


def demo_python_extractor() -> None:
    """Demonstrate Python metadata extraction."""
    print("=" * 80)
    print("PYTHON EXTRACTOR DEMONSTRATION")
    print("=" * 80)

    extractor = PythonExtractor()

    # Try to extract from cheap-py if it exists
    cheap_py_path = Path(__file__).parent.parent.parent / "cheap-py" / "src"

    if cheap_py_path.exists():
        print(f"\nExtracting from: {cheap_py_path}")
        artifacts = extractor.extract_metadata(cheap_py_path)

        print(f"\nExtracted {len(artifacts)} artifacts")

        # Group by type
        by_type: dict[str, int] = {}
        for artifact in artifacts:
            by_type[artifact.type] = by_type.get(artifact.type, 0) + 1

        print("\nArtifacts by type:")
        for artifact_type, count in sorted(by_type.items()):
            print(f"  {artifact_type}: {count}")

        # Show some examples
        print("\nExample artifacts:")
        for artifact in artifacts[:5]:
            print(f"\n  [{artifact.type}] {artifact.name}")
            print(f"    Module: {artifact.module}")
            print(f"    Description: {artifact.description[:80]}...")
            if artifact.relations:
                print(f"    Relations: {', '.join(artifact.relations)}")
            print(f"    Tags: {', '.join(artifact.tags)}")
    else:
        print(f"\nDirectory not found: {cheap_py_path}")
        print("Skipping Python extraction demo.")


def demo_typescript_extractor() -> None:
    """Demonstrate TypeScript metadata extraction."""
    print("\n" + "=" * 80)
    print("TYPESCRIPT EXTRACTOR DEMONSTRATION")
    print("=" * 80)

    extractor = TypeScriptExtractor()

    # Try to extract from cheap-ts if it exists
    cheap_ts_path = Path(__file__).parent.parent.parent / "cheap-ts" / "src"

    if cheap_ts_path.exists():
        print(f"\nExtracting from: {cheap_ts_path}")
        artifacts = extractor.extract_metadata(cheap_ts_path)

        print(f"\nExtracted {len(artifacts)} artifacts")

        # Group by type
        by_type: dict[str, int] = {}
        for artifact in artifacts:
            by_type[artifact.type] = by_type.get(artifact.type, 0) + 1

        print("\nArtifacts by type:")
        for artifact_type, count in sorted(by_type.items()):
            print(f"  {artifact_type}: {count}")

        # Show some examples
        print("\nExample artifacts:")
        for artifact in artifacts[:5]:
            print(f"\n  [{artifact.type}] {artifact.name}")
            print(f"    Module: {artifact.module}")
            print(f"    Description: {artifact.description[:80]}...")
            if artifact.relations:
                print(f"    Relations: {', '.join(artifact.relations)}")
            print(f"    Tags: {', '.join(artifact.tags)}")
    else:
        print(f"\nDirectory not found: {cheap_ts_path}")
        print("Skipping TypeScript extraction demo.")


def demo_embedding_text() -> None:
    """Demonstrate embedding text generation."""
    print("\n" + "=" * 80)
    print("EMBEDDING TEXT GENERATION DEMONSTRATION")
    print("=" * 80)

    # Create sample artifacts
    from src.extractors.base import MetadataArtifact

    # Python class example
    py_class = MetadataArtifact(
        id="python_class_example",
        name="DataCatalog",
        type="class",
        source_type="code",
        language="python",
        module="cheap.core",
        description="A catalog for managing data entities and relationships",
        relations=["extends Protocol"],
        tags=["python", "code", "class", "protocol"],
        source_file="cheap/core/catalog.py",
        source_line=10,
    )

    # TypeScript interface example
    ts_interface = MetadataArtifact(
        id="typescript_interface_example",
        name="ICatalog",
        type="interface",
        source_type="code",
        language="typescript",
        module="cheap/core",
        description="Core catalog interface for CHEAP metadata management",
        relations=["extends IEntity", "extends IHierarchical"],
        tags=["typescript", "code", "interface"],
        source_file="cheap/core/catalog.ts",
        source_line=15,
    )

    print("\nPython Class Embedding Text:")
    print("-" * 80)
    print(py_class.to_embedding_text())

    print("\n\nTypeScript Interface Embedding Text:")
    print("-" * 80)
    print(ts_interface.to_embedding_text())


def main() -> None:
    """Run all demonstrations."""
    demo_python_extractor()
    demo_typescript_extractor()
    demo_embedding_text()

    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
