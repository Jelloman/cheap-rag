"""Test Odoo PostgreSQL database connection."""

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
    """Test connection to Odoo database."""
    extractor = PostgresExtractor()

    # Get password from environment
    password = os.getenv("ODOO_DB_PASSWORD")
    if not password:
        print("[ERROR] ODOO_DB_PASSWORD not found in environment")
        print("   Make sure .env file exists and contains ODOO_DB_PASSWORD")
        return False

    print("=" * 80)
    print("Testing Odoo PostgreSQL Connection")
    print("=" * 80)

    try:
        # Connect to Odoo database
        print("\n1. Connecting to database...")
        extractor.connect({
            "host": "localhost",
            "port": 5432,
            "database": "odoo",
            "user": "odoo",
            "password": password
        })

        print("   [OK] Connected successfully!")

        # List available tables
        print("\n2. Listing available tables...")
        tables = extractor.inspector.get_table_names(schema="public")
        print(f"   [OK] Found {len(tables)} tables in 'public' schema")

        # Show eCommerce-related tables
        print("\n3. eCommerce-related tables:")
        ecommerce_tables = [
            "sale_order", "sale_order_line", "sale_order_template",
            "product_product", "product_template", "product_category",
            "stock_picking", "stock_move", "stock_location",
            "res_partner", "res_partner_category",
            "account_move", "account_move_line"
        ]

        found_tables = []
        missing_tables = []

        for table in ecommerce_tables:
            if table in tables:
                found_tables.append(table)
                print(f"   [OK] {table}")
            else:
                missing_tables.append(table)

        if missing_tables:
            print(f"\n4. Tables not found (may not be in Odoo installation):")
            for table in missing_tables:
                print(f"   [MISSING] {table}")

        # Show sample of all tables
        print(f"\n5. Sample of all available tables (first 20):")
        for table in sorted(tables)[:20]:
            print(f"   - {table}")

        if len(tables) > 20:
            print(f"   ... and {len(tables) - 20} more")

        # Disconnect
        extractor.disconnect()
        print("\n" + "=" * 80)
        print(f"Connection test successful!")
        print(f"Ready to extract from {len(found_tables)} eCommerce tables")
        print("=" * 80)

        return True

    except Exception as e:
        print(f"\n[ERROR] Connection failed: {e}")
        print("\nTroubleshooting:")
        print("1. Verify Odoo is running: systemctl status odoo (Linux) or check services")
        print("2. Verify PostgreSQL is running: systemctl status postgresql")
        print("3. Check database exists: psql -l | grep odoo")
        print("4. Test manual connection: psql -h localhost -U odoo -d odoo")
        print("5. Verify password in .env matches database password")
        return False


if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)
