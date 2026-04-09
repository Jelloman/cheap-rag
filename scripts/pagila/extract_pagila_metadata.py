"""Extract metadata from Pagila PostgreSQL database."""

import json
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
import yaml

from src.extractors.postgres_extractor import PostgresExtractor

# Load environment variables
load_dotenv()


def main():
    """Extract Pagila metadata."""
    print("=" * 80)
    print("Pagila Database Metadata Extraction")
    print("=" * 80)

    # Load configuration
    config_path = Path("config/local.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    db_config = config["indexing"]["databases"]["pagila"]

    if not db_config["enabled"]:
        print("\n[ERROR] pagila database is not enabled in config/local.yaml")
        print("Please set 'enabled: true' in the configuration.")
        return 1

    # Get password from environment
    password = os.getenv("PAGILA_DB_PASSWORD")
    if not password:
        print("\n[ERROR] PAGILA_DB_PASSWORD not found in environment")
        print("Make sure .env file exists and contains PAGILA_DB_PASSWORD")
        return 1

    # Replace environment variable in config
    connection_config = db_config["connection"].copy()
    connection_config["password"] = password

    print(f"\nConnecting to Pagila database...")
    print(f"  Host: {connection_config['host']}")
    print(f"  Port: {connection_config['port']}")
    print(f"  Database: {connection_config['database']}")
    print(f"  User: {connection_config['user']}")
    print(f"  Schema: {db_config['schema']}")

    # Create extractor
    extractor = PostgresExtractor()

    try:
        # Connect
        extractor.connect(connection_config)
        print(f"  [OK] Connected successfully!")

        # Extract schema
        print(f"\nExtracting metadata...")
        artifacts = extractor.extract_schema(db_config["schema"])

        print(f"  Extracted {len(artifacts)} artifacts before filtering")
        tables_before = [a for a in artifacts if a.type == "table"]
        print(f"  Tables found: {[t.name for t in tables_before[:10]]}")

        # Filter by include_tables if specified
        if "include_tables" in db_config and db_config["include_tables"]:
            include_tables = set(db_config["include_tables"])
            print(f"  Filtering to {len(include_tables)} specified tables...")
            print(f"  Include list: {sorted(include_tables)}")

            # Filter artifacts
            filtered = []
            for artifact in artifacts:
                # Keep table artifacts that are in include_tables
                if artifact.type == "table" and artifact.name in include_tables:
                    filtered.append(artifact)
                # Keep column, index, constraint artifacts for included tables
                elif artifact.type in ["column", "index", "constraint"]:
                    table_name = artifact.metadata.get("table_name")
                    if table_name in include_tables:
                        filtered.append(artifact)
                # Keep relationship artifacts between included tables
                elif artifact.type == "relationship":
                    from_table = artifact.metadata.get("from_table")
                    to_table = artifact.metadata.get("to_table")
                    if from_table in include_tables and to_table in include_tables:
                        filtered.append(artifact)

            artifacts = filtered

        # Add configured tags
        if "tags" in db_config:
            for artifact in artifacts:
                artifact.tags.extend(db_config["tags"])

        # Disconnect
        extractor.disconnect()

        # Group by type
        tables = [a for a in artifacts if a.type == "table"]
        columns = [a for a in artifacts if a.type == "column"]
        indexes = [a for a in artifacts if a.type == "index"]
        constraints = [a for a in artifacts if a.type == "constraint"]
        relationships = [a for a in artifacts if a.type == "relationship"]

        print(f"\n[OK] Extraction complete!")
        print(f"  Tables: {len(tables)}")
        print(f"  Columns: {len(columns)}")
        print(f"  Indexes: {len(indexes)}")
        print(f"  Constraints: {len(constraints)}")
        print(f"  Relationships: {len(relationships)}")
        print(f"  Total artifacts: {len(artifacts)}")

        # Display table summary
        print(f"\nExtracted Tables:")
        for table in tables:
            table_columns = [a for a in columns if a.metadata.get("table_name") == table.name]
            print(f"  - {table.name} ({len(table_columns)} columns)")

        # Display relationships
        if relationships:
            print(f"\nRelationships:")
            for rel in relationships:
                from_table = rel.metadata.get("from_table")
                to_table = rel.metadata.get("to_table")
                print(f"  - {from_table} -> {to_table}")

        # Save to JSON
        output_dir = Path("data/metadata")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "pagila_metadata.json"

        with open(output_path, "w") as f:
            json.dump([a.to_dict() for a in artifacts], f, indent=2)

        print(f"\n[OK] Metadata saved to: {output_path}")

        # Display some example embedding texts
        print("\n" + "=" * 80)
        print("Sample Embedding Texts")
        print("=" * 80)

        if tables:
            print(f"\nTable: {tables[0].name}")
            print("-" * 40)
            print(tables[0].to_embedding_text())

        if columns:
            print(f"\nColumn: {columns[0].name}")
            print("-" * 40)
            print(columns[0].to_embedding_text())

        if relationships:
            print(f"\nRelationship: {relationships[0].name}")
            print("-" * 40)
            print(relationships[0].to_embedding_text())

        print("\n" + "=" * 80)
        print("Extraction Complete!")
        print("=" * 80)
        print("\nNext steps:")
        print("1. Review metadata in data/metadata/pagila_metadata.json")
        print("2. Generate embeddings: python generate_embeddings.py")
        print("3. Query the schema: python scripts/query_example.py")

        return 0

    except Exception as e:
        print(f"\n[ERROR] Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
