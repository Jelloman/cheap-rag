"""Demonstration of Java code metadata extraction.

This script extracts metadata from the CHEAP core Java interfaces,
demonstrating how code artifacts are represented in the unified model.
"""

import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extractors.java_extractor import JavaExtractor


def extract_and_display_metadata(source_path: Path) -> None:
    """Extract metadata from Java source files and display results.

    Args:
        source_path: Path to Java source directory.
    """
    if not source_path.exists():
        print(f"Error: Source path does not exist: {source_path}")
        print("\nThis demo requires the cheap-core Java project.")
        print("Expected path: ../cheap-core/src/main/java")
        return

    extractor = JavaExtractor()

    # Extract metadata
    print(f"\nExtracting metadata from {source_path}...")
    artifacts = extractor.extract_metadata(source_path)

    if not artifacts:
        print("No artifacts extracted. Check that Java files exist in the source path.")
        return

    # Group artifacts by type
    interfaces = [a for a in artifacts if a.type == "interface"]
    classes = [a for a in artifacts if a.type == "class"]
    fields = [a for a in artifacts if a.type == "field"]

    print(f"\nExtraction complete!")
    print(f"  Interfaces: {len(interfaces)}")
    print(f"  Classes: {len(classes)}")
    print(f"  Fields: {len(fields)}")
    print(f"  Total artifacts: {len(artifacts)}")

    # Display interface details
    print("\n" + "=" * 80)
    print("INTERFACES (first 5)")
    print("=" * 80)
    for interface in interfaces[:5]:
        print(f"\n{interface.name}")
        print(f"  Package: {interface.module}")
        print(f"  ID: {interface.id}")
        print(f"  Description: {interface.description[:100]}..." if len(interface.description) > 100 else f"  Description: {interface.description}")
        print(f"  Tags: {', '.join(interface.tags)}")
        if interface.relations:
            print(f"  Relations: {', '.join(interface.relations)}")
        print(f"  Source: {interface.source_file}:{interface.source_line}")

    # Display class details
    if classes:
        print("\n" + "=" * 80)
        print("CLASSES (first 5)")
        print("=" * 80)
        for cls in classes[:5]:
            print(f"\n{cls.name}")
            print(f"  Package: {cls.module}")
            print(f"  Description: {cls.description[:100]}..." if len(cls.description) > 100 else f"  Description: {cls.description}")
            print(f"  Tags: {', '.join(cls.tags)}")

    # Display field examples
    if fields:
        print("\n" + "=" * 80)
        print("FIELDS (first 10)")
        print("=" * 80)
        for field in fields[:10]:
            print(f"\n{field.name}")
            print(f"  Type: {field.type}")
            print(f"  Module: {field.module}")
            print(f"  Field Type: {field.metadata.get('field_type', 'N/A')}")
            if field.constraints:
                print(f"  Constraints: {', '.join(field.constraints)}")

    # Display embedding text examples
    print("\n" + "=" * 80)
    print("EMBEDDING TEXT EXAMPLES")
    print("=" * 80)

    if interfaces:
        print("\nInterface embedding text:")
        print("-" * 40)
        print(interfaces[0].to_embedding_text())

    if classes:
        print("\nClass embedding text:")
        print("-" * 40)
        print(classes[0].to_embedding_text())

    if fields:
        print("\nField embedding text:")
        print("-" * 40)
        print(fields[0].to_embedding_text())

    # Save to JSON
    output_path = Path("./data/metadata/demo_java_metadata.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump([a.to_dict() for a in artifacts], f, indent=2)

    print(f"\n\nMetadata saved to: {output_path}")


def main():
    """Run the demo."""
    # Default to cheap-core source directory
    source_path = Path("../cheap/cheap-core/src/main/java")

    print("=" * 80)
    print("Java Code Metadata Extraction Demo")
    print("=" * 80)

    # Extract and display metadata
    extract_and_display_metadata(source_path)

    print("\n" + "=" * 80)
    print("Demo complete!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Review the extracted metadata in data/metadata/demo_java_metadata.json")
    print("2. Notice how code artifacts (interfaces, classes, fields) are")
    print("   represented in the unified MetadataArtifact model")
    print("3. Compare with the database artifacts from the SQLite demo")
    print("4. Both use the same MetadataArtifact model with source_type field")
    print("   to distinguish between 'database' and 'code' artifacts")


if __name__ == "__main__":
    main()
