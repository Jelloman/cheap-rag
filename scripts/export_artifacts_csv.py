"""Export all ChromaDB artifacts to a CSV file for gold dataset reference.

Usage:
    python scripts/export_artifacts_csv.py
    python scripts/export_artifacts_csv.py --output artifacts.csv
    python scripts/export_artifacts_csv.py --language postgresql
    python scripts/export_artifacts_csv.py --type table
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb
from chromadb.config import Settings

from src.evaluation.gold_dataset import compute_artifact_component

VECTOR_DB_PATH = "./data/vector_db"
COLLECTION_NAME = "cheap_metadata_v1"
DEFAULT_OUTPUT = "./data/artifacts.csv"

# Columns to export, in order
CSV_COLUMNS = [
    "id",
    "name",
    "component",
    "type",
    "language",
    "module",
    "source_file",
    "source_line",
    "table_name",
    "column_type",
    "nullable",
    "primary_key",
    "foreign_key",
    "from_table",
    "to_table",
    "cardinality",
    "tags",
    "description",
]

def get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(
        path=VECTOR_DB_PATH,
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_collection(name=COLLECTION_NAME)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ChromaDB artifacts to CSV")
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT, help=f"Output CSV path (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--language", "-l", help="Filter by language (e.g. postgresql, java)")
    parser.add_argument("--type", "-t", dest="artifact_type", help="Filter by type (e.g. table, column, class)")
    args = parser.parse_args()

    try:
        collection = get_collection()
        total = collection.count()
        print(f"Connected to ChromaDB: {VECTOR_DB_PATH}  collection={COLLECTION_NAME}  count={total}")
    except Exception as e:
        print(f"Failed to open ChromaDB: {e}")
        print(f"Make sure {VECTOR_DB_PATH} exists and has been indexed (run scripts/index_metadata.py first).")
        sys.exit(1)

    where: dict[str, object] | None = None
    conditions: list[dict[str, object]] = []
    if args.language:
        conditions.append({"language": args.language})
    if args.artifact_type:
        conditions.append({"type": args.artifact_type})
    if len(conditions) == 1:
        where = conditions[0]
    elif len(conditions) > 1:
        where = {"$and": conditions}

    kwargs: dict[str, object] = {"include": ["metadatas"], "limit": total}
    if where:
        kwargs["where"] = where

    results = collection.get(**kwargs)  # type: ignore[arg-type]
    artifact_ids: list[str] = results["ids"]  # type: ignore[assignment]
    metadatas: list[dict[str, object]] = results["metadatas"] or []  # type: ignore[assignment]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for aid, meta in zip(artifact_ids, metadatas):
        row: dict[str, object] = {"id": aid}
        for col in CSV_COLUMNS[1:]:
            val = meta.get(col, "")
            # Blank out sentinel "None" strings
            row[col] = "" if val in (None, "None") else val
        row["component"] = compute_artifact_component(row)
        rows.append(row)

    rows.sort(key=lambda r: (str(r.get("language", "")), str(r.get("type", "")), str(r.get("module", "")), str(r.get("component", "")), str(r.get("name", ""))))

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    filter_desc = []
    if args.language:
        filter_desc.append(f"language={args.language}")
    if args.artifact_type:
        filter_desc.append(f"type={args.artifact_type}")
    filter_str = f" (filtered: {', '.join(filter_desc)})" if filter_desc else ""
    print(f"Exported {len(artifact_ids)} artifacts{filter_str} to {output_path}")


if __name__ == "__main__":
    main()
