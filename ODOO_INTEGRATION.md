# Odoo Database Integration Guide

Quick reference for integrating the CHEAP RAG system with an Odoo PostgreSQL database.

## Prerequisites

- Odoo installed and running
- PostgreSQL database accessible
- Database credentials available
- `sqlalchemy` and `psycopg2-binary` installed (already in `requirements.txt`)

## Configuration Steps

### 1. Set Database Password

Add your Odoo database password to `.env`:

```bash
# Copy example if you haven't already
cp .env.example .env

# Edit .env and set:
ODOO_DB_PASSWORD=your_actual_odoo_password
```

### 2. Enable Odoo Database in Config

Edit `config/local.yaml` and enable the Odoo database:

```yaml
indexing:
  databases:
    odoo_ecommerce:
      enabled: true  # Change from false to true
      type: "postgresql"
      connection:
        host: "localhost"  # Update if Odoo is on different host
        port: 5432
        database: "odoo"  # Update if your database has different name
        user: "odoo"      # Update if different user
        password: "${ODOO_DB_PASSWORD}"
      schema: "public"
      # Optional: Limit extraction to specific tables
      include_tables:
        - "sale_order"
        - "sale_order_line"
        - "product_product"
        - "product_template"
        - "product_category"
        - "stock_picking"
        - "stock_move"
        - "stock_location"
        - "res_partner"
        - "res_partner_category"
        - "account_move"
        - "account_move_line"
      tags: ["odoo", "ecommerce", "sales"]

  extractors:
    postgresql:
      enabled: true  # Enable PostgreSQL extractor
```

**Note:** If you omit `include_tables`, the extractor will extract **all** tables in the schema. This may result in hundreds of tables. For Phase 1, it's recommended to limit to eCommerce-related tables.

### 3. Verify Database Connection

Test the connection with a simple script:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.extractors.postgres_extractor import PostgresExtractor

extractor = PostgresExtractor()

try:
    extractor.connect({
        "host": "localhost",
        "port": 5432,
        "database": "odoo",
        "user": "odoo",
        "password": "your_password_here"  # Use actual password
    })

    print("✓ Connected to Odoo database successfully!")

    # List available tables
    tables = extractor.inspector.get_table_names(schema="public")
    print(f"\nFound {len(tables)} tables in 'public' schema")
    print("\nFirst 10 tables:")
    for table in tables[:10]:
        print(f"  - {table}")

    extractor.disconnect()

except Exception as e:
    print(f"✗ Connection failed: {e}")
```

Save as `test_odoo_connection.py` and run:

```bash
python test_odoo_connection.py
```

### 4. Extract Metadata

Once connected successfully, extract metadata:

```bash
# Extract from Odoo database only
python scripts/index_metadata.py --databases odoo_ecommerce

# Or extract from all enabled databases
python scripts/index_metadata.py --databases all
```

Expected output:
```
Extracting from database: odoo_ecommerce
Connected to PostgreSQL database: odoo
Extracting schema: public
  Tables: 20-30
  Columns: 200-400
  Indexes: 50-100
  Relationships: 30-60
Total artifacts: 300-600

Metadata saved to: data/metadata/odoo_ecommerce_metadata.json
```

### 5. Review Extracted Metadata

```bash
# View metadata file
cat data/metadata/odoo_ecommerce_metadata.json | python -m json.tool | less

# Count artifacts by type
python -c "
import json
data = json.load(open('data/metadata/odoo_ecommerce_metadata.json'))
from collections import Counter
types = Counter(a['type'] for a in data)
for t, count in types.items():
    print(f'{t}: {count}')
"
```

Expected output:
```
table: 25
column: 350
relationship: 45
index: 80
```

## Sample Queries (After Full RAG Pipeline Complete)

Once the RAG pipeline is implemented (Week 2), you'll be able to query the Odoo schema:

### Schema Discovery
- "What tables are related to sales orders?"
- "List all tables that reference the res_partner table"
- "Show me the structure of the product_product table"

### Column Information
- "What columns in sale_order reference res_partner?"
- "Which tables have a 'company_id' column?"
- "What's the data type of the 'date_order' column in sale_order?"

### Relationships
- "How is sale_order connected to stock_picking?"
- "What foreign keys point to the product_template table?"
- "Describe the relationship between sale_order and account_move"

### Business Logic
- "How are products linked to inventory movements?"
- "What's the connection between sales orders and invoices?"
- "Which tables are involved in the order fulfillment process?"

## Troubleshooting

### Connection Refused
**Error:** `could not connect to server: Connection refused`

**Solutions:**
- Verify PostgreSQL is running: `systemctl status postgresql` (Linux) or check services (Windows)
- Check `pg_hba.conf` allows connections from localhost
- Verify port 5432 is not blocked by firewall

### Authentication Failed
**Error:** `FATAL: password authentication failed for user "odoo"`

**Solutions:**
- Verify password in `.env` matches Odoo database password
- Check database user has read permissions: `GRANT SELECT ON ALL TABLES IN SCHEMA public TO odoo;`
- Try connecting with `psql`: `psql -h localhost -U odoo -d odoo`

### Permission Denied
**Error:** `permission denied for schema public`

**Solutions:**
- Grant read permissions to odoo user:
  ```sql
  GRANT USAGE ON SCHEMA public TO odoo;
  GRANT SELECT ON ALL TABLES IN SCHEMA public TO odoo;
  ```

### Too Many Tables Extracted
**Issue:** Extracted 500+ tables, including internal Odoo tables

**Solution:**
- Add `include_tables` list to config to limit extraction to relevant tables
- Or use `exclude_patterns` (not yet implemented) to skip internal tables

## Advanced Configuration

### Extract from Multiple Odoo Modules

```yaml
databases:
  odoo_sales:
    enabled: true
    type: "postgresql"
    connection:
      database: "odoo"
      # ... other settings
    include_tables:
      - "sale_order"
      - "sale_order_line"
    tags: ["odoo", "sales"]

  odoo_inventory:
    enabled: true
    type: "postgresql"
    connection:
      database: "odoo"
      # ... other settings
    include_tables:
      - "stock_picking"
      - "stock_move"
      - "stock_location"
    tags: ["odoo", "inventory"]
```

### Extract from Remote Odoo Instance

```yaml
databases:
  odoo_production:
    enabled: true
    type: "postgresql"
    connection:
      host: "odoo.example.com"
      port: 5432
      database: "odoo_prod"
      user: "readonly_user"
      password: "${ODOO_PROD_PASSWORD}"
    schema: "public"
    tags: ["odoo", "production"]
```

## Performance Tips

### Large Schemas
- Use `include_tables` to limit extraction to relevant tables
- Extract in batches if needed (create multiple database configs)
- Consider extracting indexes and constraints separately if schema is very large

### Slow Extraction
- PostgreSQL Inspector is generally fast (<5 seconds for 100 tables)
- If slow, check database performance and network latency
- Consider caching extracted metadata for repeated queries

### Memory Usage
- Each artifact is ~1-5KB in JSON
- 500 tables × 20 columns × 2KB ≈ 20MB metadata
- Embeddings will add ~3KB per artifact (768 dims × 4 bytes)
- Total: ~50MB for large Odoo schema (acceptable)

## Next Steps

After extracting Odoo metadata:

1. **Generate Embeddings** (Week 2, Day 1)
   ```bash
   python scripts/generate_embeddings.py --metadata odoo_ecommerce
   ```

2. **Index in ChromaDB** (Week 2, Day 2)
   ```bash
   python scripts/index_vectors.py --metadata odoo_ecommerce
   ```

3. **Query the Schema** (Week 2, Day 3+)
   ```bash
   python scripts/query_example.py
   # Query: "What tables are related to sales orders?"
   ```

4. **Compare with Code Metadata** (Week 2, Day 7)
   ```bash
   # Query: "Compare the Catalog interface in Java to the sale_order table in Odoo"
   ```

## Support

If you encounter issues:
1. Check `logs/cheap-rag.log` for detailed error messages
2. Verify database connection with `psql` or database client
3. Test extraction with SQLite demo first: `python scripts/demo_sqlite_extraction.py`
4. Review PostgreSQL extractor code: `src/extractors/postgres_extractor.py`
