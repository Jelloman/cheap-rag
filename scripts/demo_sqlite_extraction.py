"""Demonstration of database metadata extraction using SQLite.

This script creates a simple SQLite database with a schema similar to
a simplified eCommerce system, then extracts metadata using the SQLite extractor.

Run this to verify database extraction works before connecting to Odoo.
"""

import json
import sqlite3
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extractors.sqlite_extractor import SqliteExtractor


def create_demo_database(db_path: str) -> None:
    """Create a demo SQLite database with eCommerce-like schema.

    Args:
        db_path: Path to SQLite database file.
    """
    # Remove existing database
    if Path(db_path).exists():
        Path(db_path).unlink()

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create customers table
    cursor.execute("""
        CREATE TABLE customers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create products table
    cursor.execute("""
        CREATE TABLE products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            description TEXT,
            price REAL NOT NULL,
            stock_quantity INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create orders table
    cursor.execute("""
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id INTEGER NOT NULL,
            order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            total_amount REAL NOT NULL,
            status TEXT CHECK(status IN ('pending', 'paid', 'shipped', 'delivered')),
            FOREIGN KEY (customer_id) REFERENCES customers(id)
        )
    """)

    # Create order_items table
    cursor.execute("""
        CREATE TABLE order_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id INTEGER NOT NULL,
            product_id INTEGER NOT NULL,
            quantity INTEGER NOT NULL,
            unit_price REAL NOT NULL,
            FOREIGN KEY (order_id) REFERENCES orders(id),
            FOREIGN KEY (product_id) REFERENCES products(id)
        )
    """)

    # Create indexes
    cursor.execute("CREATE INDEX idx_customers_email ON customers(email)")
    cursor.execute("CREATE INDEX idx_orders_customer ON orders(customer_id)")
    cursor.execute("CREATE INDEX idx_order_items_order ON order_items(order_id)")
    cursor.execute("CREATE INDEX idx_order_items_product ON order_items(product_id)")

    conn.commit()
    conn.close()

    print(f"Created demo database at: {db_path}")


def extract_and_display_metadata(db_path: str) -> None:
    """Extract metadata from SQLite database and display results.

    Args:
        db_path: Path to SQLite database file.
    """
    extractor = SqliteExtractor()

    # Connect to database
    extractor.connect({"path": db_path})

    # Extract schema
    print(f"\nExtracting schema from {db_path}...")
    artifacts = extractor.extract_schema()

    # Disconnect
    extractor.disconnect()

    # Group artifacts by type
    tables = [a for a in artifacts if a.type == "table"]
    columns = [a for a in artifacts if a.type == "column"]
    indexes = [a for a in artifacts if a.type == "index"]
    relationships = [a for a in artifacts if a.type == "relationship"]

    print(f"\nExtraction complete!")
    print(f"  Tables: {len(tables)}")
    print(f"  Columns: {len(columns)}")
    print(f"  Indexes: {len(indexes)}")
    print(f"  Relationships: {len(relationships)}")

    # Display table details
    print("\n" + "=" * 80)
    print("TABLES")
    print("=" * 80)
    for table in tables:
        print(f"\n{table.name}")
        print(f"  ID: {table.id}")
        print(f"  Description: {table.description}")
        print(f"  Tags: {', '.join(table.tags)}")

    # Display some column examples
    print("\n" + "=" * 80)
    print("COLUMNS (first 10)")
    print("=" * 80)
    for column in columns[:10]:
        print(f"\n{column.module}.{column.name}")
        print(f"  Type: {column.metadata.get('column_type', 'N/A')}")
        print(f"  Nullable: {column.metadata.get('nullable', True)}")
        if column.metadata.get('primary_key'):
            print(f"  PRIMARY KEY")
        if column.metadata.get('foreign_key'):
            print(f"  Foreign Key to: {column.metadata['foreign_key']}")

    # Display relationships
    print("\n" + "=" * 80)
    print("RELATIONSHIPS")
    print("=" * 80)
    for rel in relationships:
        print(f"\n{rel.name}")
        print(f"  {rel.metadata.get('from_table')} -> {rel.metadata.get('to_table')}")
        print(f"  Columns: {', '.join(rel.relations)}")

    # Display embedding text examples
    print("\n" + "=" * 80)
    print("EMBEDDING TEXT EXAMPLES")
    print("=" * 80)

    if tables:
        print("\nTable embedding text:")
        print("-" * 40)
        print(tables[0].to_embedding_text())

    if columns:
        print("\nColumn embedding text:")
        print("-" * 40)
        print(columns[0].to_embedding_text())

    if relationships:
        print("\nRelationship embedding text:")
        print("-" * 40)
        print(relationships[0].to_embedding_text())

    # Save to JSON
    output_path = Path("./data/metadata/demo_sqlite_metadata.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump([a.to_dict() for a in artifacts], f, indent=2)

    print(f"\n\nMetadata saved to: {output_path}")


def main():
    """Run the demo."""
    db_path = "./data/demo_ecommerce.db"

    print("=" * 80)
    print("SQLite Database Metadata Extraction Demo")
    print("=" * 80)

    # Create demo database
    create_demo_database(db_path)

    # Extract and display metadata
    extract_and_display_metadata(db_path)

    print("\n" + "=" * 80)
    print("Demo complete!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Review the extracted metadata in data/metadata/demo_sqlite_metadata.json")
    print("2. Notice how database artifacts (tables, columns, relationships) are")
    print("   represented in the unified MetadataArtifact model")
    print("3. When Odoo is installed, configure config/local.yaml and extract")
    print("   metadata from the Odoo PostgreSQL database using PostgresExtractor")


if __name__ == "__main__":
    main()
