

# SCRIPT: browse_chroma.md

## Look up specific IDs from the gold dataset
```bash
python scripts/browse_chroma.py ids pg_table_e5f19fb863d1fa62 pg_column_bc27f24fe2fd21d7
```

## Add --doc to also see the embedding text stored with each artifact
```bash
python scripts/browse_chroma.py ids pg_table_e5f19fb863d1fa62 --doc
```

## Browse the gold dataset interactively
Shows each query, its selected relevant artifacts, and the remaining candidates (with similarity scores) so you can judge whether the selection was right.
```bash
python scripts/browse_chroma.py gold
```

# Review one specific query
```bash
python scripts/browse_chroma.py gold entity_001
```

# Dump all queries without pausing (good for piping to a file)
```bash
python scripts/browse_chroma.py gold --no-interactive > review.txt
```

# List artifacts, filter by language/type
```bash
python scripts/browse_chroma.py list --language postgresql --type table
python scripts/browse_chroma.py list --language postgresql --type column --limit 50
```

# Find artifacts by name or description substring
```bash
python scripts/browse_chroma.py search film
python scripts/browse_chroma.py search customer
```

The gold command is probably what you'll use most. For each query it shows:
- The query text, category, difficulty
- The currently-selected relevant artifacts (green, with similarity score) — these are what you're validating
- The remaining candidates (yellow) that were retrieved but not marked relevant — so you can promote/demote as needed
