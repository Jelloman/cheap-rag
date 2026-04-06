# Pagila Database Integration Guide

Quick reference for integrating the CHEAP RAG system with the Pagila sample database (PostgreSQL).

Pagila is a port of the MySQL Sakila sample database to PostgreSQL. It models a DVD rental store with actors, films, customers, rentals, payments, and inventory. It is well-suited as a reference database because it has a manageable number of tables (15 core), rich relationships, and realistic sample data.

## Prerequisites

- Docker with Pagila container running (`pagila/docker-compose.yml`)
- PostgreSQL accessible on localhost:5432
- `sqlalchemy` and `psycopg2-binary` installed (already in `requirements.txt`)

## Pagila Schema Overview

### Core Tables (15)

| Table | Description |
|-------|-------------|
| `actor` | Film actors (first_name, last_name) |
| `film` | Films with title, description, release year, rating, rental info |
| `film_actor` | Many-to-many: films ↔ actors |
| `film_category` | Many-to-many: films ↔ categories |
| `category` | Film categories (Action, Comedy, Drama, etc.) |
| `language` | Languages for films |
| `customer` | Rental customers with store association |
| `address` | Addresses (shared by customers, staff, stores) |
| `city` | Cities |
| `country` | Countries |
| `inventory` | Physical film copies at stores |
| `rental` | Rental transactions (customer, inventory, dates) |
| `payment` | Payments for rentals |
| `staff` | Store employees |
| `store` | Rental store locations |

### Key Relationships

- `customer` → `address` → `city` → `country` (geographic hierarchy)
- `rental` → `customer`, `inventory`, `staff` (who rented what, processed by whom)
- `payment` → `customer`, `staff`, `rental` (payment for a rental)
- `inventory` → `film`, `store` (which film copies at which store)
- `film` → `language` (original language)
- `film_actor` → `film`, `actor` (many-to-many)
- `film_category` → `film`, `category` (many-to-many)
- `store` → `address`, `staff` (manager)
- `staff` → `address`, `store`

## Configuration Steps

### 1. Set Database Password

Add your Pagila database password to `.env`:

```bash
# Copy example if you haven't already
cp .env.example .env

# Edit .env and set:
PAGILA_DB_PASSWORD=123456
```

### 2. Enable Pagila Database in Config

The Pagila database is already enabled in `config/local.yaml`:

```yaml
indexing:
  databases:
    pagila:
      enabled: true
      type: "postgresql"
      connection:
        host: "localhost"
        port: 5432
        database: "postgres"
        user: "postgres"
        password: "${PAGILA_DB_PASSWORD}"
      schema: "public"
      include_tables:
        - "actor"
        - "film"
        - "film_actor"
        - "film_category"
        - "category"
        - "language"
        - "customer"
        - "address"
        - "city"
        - "country"
        - "inventory"
        - "rental"
        - "payment"
        - "staff"
        - "store"
      tags: ["pagila", "dvd-rental", "sample"]
```

### 3. Verify Database Connection

```bash
python test_pagila_connection.py
```

### 4. Extract Metadata

```bash
# Extract from Pagila database
python extract_pagila_metadata.py
```

Expected output:
```
Extracting from database: pagila
Connected to PostgreSQL database: postgres
Extracting schema: public
  Tables: 15
  Columns: ~70
  Indexes: ~20
  Relationships: ~15
Total artifacts: ~120

Metadata saved to: data/metadata/pagila_metadata.json
```

### 5. Generate Embeddings and Index

```bash
python generate_embeddings.py
```

## Sample Queries

Once the RAG pipeline is running with Pagila data indexed:

### Schema Discovery
- "What tables are in the Pagila database?"
- "Show me the structure of the film table"
- "What tables are related to rentals?"

### Column Information
- "What columns does the customer table have?"
- "What is the data type of the rental_rate column in film?"
- "Which columns in the film table are required (not nullable)?"

### Relationships
- "How is the rental table connected to customer?"
- "What foreign keys point to the film table?"
- "Describe the relationship between inventory and store"

### Business Logic
- "How are films linked to actors?"
- "What's the connection between rentals and payments?"
- "How is the geographic hierarchy (country → city → address) structured?"

## Troubleshooting

### Connection Refused
**Error:** `could not connect to server: Connection refused`

**Solutions:**
- Verify Pagila container is running: `docker ps | grep pagila`
- Start the container: `cd pagila && docker compose up -d`
- Verify port 5432 is not blocked or used by another service

### Authentication Failed
**Error:** `FATAL: password authentication failed for user "postgres"`

**Solutions:**
- Verify password in `.env` matches `docker-compose.yml` (default: `123456`)
- Test manual connection: `psql -h localhost -U postgres -d postgres`

## Performance Notes

- Pagila is a small database (~15 core tables, ~70 columns)
- Extraction completes in under 1 second
- Total metadata size: ~50-100KB JSON
- Embeddings: ~500KB (768 dims × ~120 artifacts × 4 bytes)
- Ideal for development, testing, and evaluation dataset creation
