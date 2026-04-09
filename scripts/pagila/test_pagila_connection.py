"""Test Pagila PostgreSQL database connection."""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from src.extractors.postgres_extractor import PostgresExtractor

# Load environment variables
load_dotenv()

def test_connection():
    """Test connection to Pagila database."""
    extractor = PostgresExtractor()

    # Get password from environment
    password = os.getenv("PAGILA_DB_PASSWORD")
    if not password:
        print("[ERROR] PAGILA_DB_PASSWORD not found in environment")
        print("   Make sure .env file exists and contains PAGILA_DB_PASSWORD")
        return False

    print("=" * 80)
    print("Testing Pagila PostgreSQL Connection")
    print("=" * 80)

    try:
        # Connect to Pagila database
        print("\n1. Connecting to database...")
        extractor.connect({
            "host": "localhost",
            "port": 5432,
            "database": "postgres",
            "user": "postgres",
            "password": password
        })

        print("   [OK] Connected successfully!")

        # List available tables
        print("\n2. Listing available tables...")
        tables = extractor.inspector.get_table_names(schema="public")
        print(f"   [OK] Found {len(tables)} tables in 'public' schema")

        # Show expected Pagila core tables
        print("\n3. Core Pagila tables:")
        core_tables = [
            "actor", "film", "film_actor", "film_category",
            "category", "language", "customer", "address",
            "city", "country", "inventory", "rental",
            "payment", "staff", "store"
        ]

        found_tables = []
        missing_tables = []

        for table in core_tables:
            if table in tables:
                found_tables.append(table)
                print(f"   [OK] {table}")
            else:
                missing_tables.append(table)

        if missing_tables:
            print(f"\n4. Tables not found:")
            for table in missing_tables:
                print(f"   [MISSING] {table}")

        # Show all tables
        print(f"\n5. All available tables:")
        for table in sorted(tables):
            print(f"   - {table}")

        # Disconnect
        extractor.disconnect()
        print("\n" + "=" * 80)
        print(f"Connection test successful!")
        print(f"Ready to extract from {len(found_tables)} core tables")
        print("=" * 80)

        return True

    except Exception as e:
        print(f"\n[ERROR] Connection failed: {e}")
        print("\nTroubleshooting:")
        print("1. Verify Pagila Docker container is running: docker ps | grep pagila")
        print("2. Check port 5432 is accessible: docker port pagila")
        print("3. Verify password in .env matches docker-compose.yml")
        print("4. Test manual connection: psql -h localhost -U postgres -d postgres")
        return False


if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)
