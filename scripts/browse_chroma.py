"""Browse and inspect ChromaDB artifacts for manual gold dataset review.

Usage:
    # Look up one or more artifact IDs
    python scripts/browse_chroma.py ids pg_table_e5f19fb863d1fa62 pg_column_bc27f24fe2fd21d7

    # Review all queries in the gold dataset (interactive)
    python scripts/browse_chroma.py gold

    # Review a specific gold dataset query by ID
    python scripts/browse_chroma.py gold entity_001

    # List artifacts with optional filters
    python scripts/browse_chroma.py list
    python scripts/browse_chroma.py list --language postgresql --type table --limit 20

    # Search by name substring
    python scripts/browse_chroma.py search film
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import chromadb
from chromadb.config import Settings

VECTOR_DB_PATH = "./data/vector_db"
COLLECTION_NAME = "cheap_metadata_v1"
GOLD_DATASET_PATH = "./tests/fixtures/gold_dataset.json"

# ANSI colors for terminal output
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
RED = "\033[31m"
RESET = "\033[0m"


def get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(
        path=VECTOR_DB_PATH,
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_collection(name=COLLECTION_NAME)


def format_artifact(artifact_id: str, metadata: dict[str, object], document: str | None = None, score: float | None = None) -> str:
    """Format a single artifact for display."""
    lines: list[str] = []

    score_str = f"  {DIM}score={score:.4f}{RESET}" if score is not None else ""
    lines.append(f"{BOLD}{CYAN}{artifact_id}{RESET}{score_str}")

    name = metadata.get("name", "?")
    atype = metadata.get("type", "?")
    lang = metadata.get("language", "?")
    module = metadata.get("module", "?")
    lines.append(f"  {BOLD}{name}{RESET}  [{atype}]  lang={lang}  module={module}")

    desc = metadata.get("description", "")
    if desc:
        # Wrap long descriptions
        lines.append(f"  {desc[:200]}{'...' if len(str(desc)) > 200 else ''}")

    # Extra fields
    extra_keys = ["table_name", "column_type", "nullable", "primary_key", "foreign_key", "from_table", "to_table", "cardinality", "source_file", "source_line", "tags"]
    extras = {k: metadata[k] for k in extra_keys if k in metadata and metadata[k] not in (None, "", "None")}
    if extras:
        parts = [f"{k}={v}" for k, v in extras.items()]
        lines.append(f"  {DIM}{', '.join(parts)}{RESET}")

    if document:
        lines.append(f"  {DIM}--- embedding text ---{RESET}")
        lines.append(f"  {DIM}{document[:300]}{'...' if len(document) > 300 else ''}{RESET}")

    return "\n".join(lines)


def cmd_ids(collection: chromadb.Collection, ids: list[str], show_document: bool = False) -> None:
    """Look up and display specific artifact IDs."""
    results = collection.get(
        ids=ids,
        include=["metadatas", "documents"],
    )

    found_ids: list[str] = results["ids"]  # type: ignore[assignment]
    metadatas: list[dict[str, object]] = results["metadatas"] or []  # type: ignore[assignment]
    documents: list[str] = results["documents"] or []  # type: ignore[assignment]

    found_set = set(found_ids)
    for aid in ids:
        if aid not in found_set:
            print(f"{RED}NOT FOUND: {aid}{RESET}")

    for aid, meta, doc in zip(found_ids, metadatas, documents):
        print(format_artifact(aid, meta, doc if show_document else None))
        print()


def cmd_list(collection: chromadb.Collection, language: str | None, artifact_type: str | None, limit: int) -> None:
    """List artifacts with optional filters, paginated."""
    where: dict[str, object] | None = None
    conditions: list[dict[str, object]] = []
    if language:
        conditions.append({"language": language})
    if artifact_type:
        conditions.append({"type": artifact_type})
    if len(conditions) == 1:
        where = conditions[0]
    elif len(conditions) > 1:
        where = {"$and": conditions}

    kwargs: dict[str, object] = {
        "limit": limit,
        "include": ["metadatas"],
    }
    if where:
        kwargs["where"] = where

    results = collection.get(**kwargs)  # type: ignore[arg-type]

    found_ids: list[str] = results["ids"]  # type: ignore[assignment]
    metadatas: list[dict[str, object]] = results["metadatas"] or []  # type: ignore[assignment]

    total = collection.count()
    filter_desc = []
    if language:
        filter_desc.append(f"language={language}")
    if artifact_type:
        filter_desc.append(f"type={artifact_type}")
    filter_str = f" ({', '.join(filter_desc)})" if filter_desc else ""
    print(f"{BOLD}Showing {len(found_ids)} of {total} total artifacts{filter_str}{RESET}\n")

    for aid, meta in zip(found_ids, metadatas):
        print(format_artifact(aid, meta))
        print()


def cmd_search(collection: chromadb.Collection, name_substring: str, limit: int = 20) -> None:
    """Search artifacts by name substring (case-insensitive)."""
    # Fetch all and filter client-side — ChromaDB doesn't support substring matching
    results = collection.get(include=["metadatas"])
    found_ids: list[str] = results["ids"]  # type: ignore[assignment]
    metadatas: list[dict[str, object]] = results["metadatas"] or []  # type: ignore[assignment]

    matches = [
        (aid, meta)
        for aid, meta in zip(found_ids, metadatas)
        if name_substring.lower() in str(meta.get("name", "")).lower()
        or name_substring.lower() in str(meta.get("description", "")).lower()
    ]

    print(f"{BOLD}Found {len(matches)} artifacts matching '{name_substring}'{RESET}\n")
    for aid, meta in matches[:limit]:
        print(format_artifact(aid, meta))
        print()
    if len(matches) > limit:
        print(f"{DIM}... and {len(matches) - limit} more (increase --limit to see all){RESET}")


def cmd_gold(collection: chromadb.Collection, query_id: str | None, interactive: bool) -> None:
    """Review gold dataset queries and their referenced artifacts."""
    gold_path = Path(GOLD_DATASET_PATH)
    if not gold_path.exists():
        print(f"{RED}Gold dataset not found: {gold_path}{RESET}")
        sys.exit(1)

    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    queries: list[dict[str, object]] = gold["queries"]

    if query_id:
        queries = [q for q in queries if q["id"] == query_id]
        if not queries:
            print(f"{RED}Query ID '{query_id}' not found in gold dataset{RESET}")
            sys.exit(1)

    total = len(queries)
    for idx, query in enumerate(queries):
        qid = query["id"]
        category = query.get("category", "?")
        q_text = query["query"]
        language = query.get("language", "?")
        difficulty = query.get("difficulty", "?")
        notes = query.get("notes", "")
        relevant_ids: list[str] = query.get("relevant_artifact_ids", [])  # type: ignore[assignment]
        all_candidates: list[str] = query.get("metadata", {}).get("all_candidates", [])  # type: ignore[assignment,union-attr]
        candidate_scores: list[float] = query.get("metadata", {}).get("candidate_scores", [])  # type: ignore[assignment,union-attr]

        # Header
        print(f"\n{'='*70}")
        print(f"{BOLD}Query {idx+1}/{total}: {qid}{RESET}  [{category}]  difficulty={difficulty}  lang={language}")
        print(f"{BOLD}Q: {q_text}{RESET}")
        if notes:
            print(f"{DIM}Notes: {notes}{RESET}")
        print()

        # Fetch all candidates (for context), display relevant ones prominently
        fetch_ids = list(dict.fromkeys(relevant_ids + [c for c in all_candidates if c not in relevant_ids]))
        if not fetch_ids:
            print(f"{YELLOW}No artifact IDs listed for this query.{RESET}")
        else:
            results = collection.get(ids=fetch_ids, include=["metadatas", "documents"])
            found_ids: list[str] = results["ids"]  # type: ignore[assignment]
            found_meta: list[dict[str, object]] = results["metadatas"] or []  # type: ignore[assignment]
            found_docs: list[str] = results["documents"] or []  # type: ignore[assignment]
            id_to_data = {aid: (meta, doc) for aid, meta, doc in zip(found_ids, found_meta, found_docs)}

            score_map = dict(zip(all_candidates, candidate_scores))

            print(f"{GREEN}{BOLD}Relevant artifacts ({len(relevant_ids)} selected):{RESET}")
            for rid in relevant_ids:
                if rid in id_to_data:
                    meta, doc = id_to_data[rid]
                    score = score_map.get(rid)
                    print(format_artifact(rid, meta, score=score))
                else:
                    print(f"  {RED}NOT IN DB: {rid}{RESET}")
                print()

            # Show remaining candidates not in relevant set
            remaining = [c for c in all_candidates if c not in relevant_ids]
            if remaining:
                print(f"{YELLOW}Other candidates (not marked relevant):{RESET}")
                for cid in remaining:
                    if cid in id_to_data:
                        meta, doc = id_to_data[cid]
                        score = score_map.get(cid)
                        print(format_artifact(cid, meta, score=score))
                    else:
                        print(f"  {RED}NOT IN DB: {cid}{RESET}")
                    print()

        if interactive and idx < total - 1:
            try:
                resp = input(f"{DIM}Press Enter for next query, 'q' to quit: {RESET}").strip().lower()
                if resp == "q":
                    break
            except (EOFError, KeyboardInterrupt):
                break

    print(f"\n{BOLD}Done. Reviewed {min(idx+1, total)}/{total} queries.{RESET}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Browse ChromaDB artifacts")
    sub = parser.add_subparsers(dest="command", required=True)

    # ids subcommand
    p_ids = sub.add_parser("ids", help="Look up artifact(s) by ID")
    p_ids.add_argument("artifact_ids", nargs="+", metavar="ID")
    p_ids.add_argument("--doc", action="store_true", help="Show embedding document text")

    # list subcommand
    p_list = sub.add_parser("list", help="List artifacts with optional filters")
    p_list.add_argument("--language", "-l", help="Filter by language (e.g. postgresql, java)")
    p_list.add_argument("--type", "-t", dest="artifact_type", help="Filter by type (e.g. table, column, class)")
    p_list.add_argument("--limit", "-n", type=int, default=20, help="Max results (default 20)")

    # search subcommand
    p_search = sub.add_parser("search", help="Search artifacts by name/description substring")
    p_search.add_argument("query", help="Substring to search for in name or description")
    p_search.add_argument("--limit", "-n", type=int, default=20)

    # gold subcommand
    p_gold = sub.add_parser("gold", help="Review gold dataset queries and their artifacts")
    p_gold.add_argument("query_id", nargs="?", help="Show only this query ID (e.g. entity_001)")
    p_gold.add_argument("--no-interactive", action="store_true", help="Dump all queries without pausing")

    args = parser.parse_args()

    try:
        collection = get_collection()
        print(f"{DIM}Connected to ChromaDB: {VECTOR_DB_PATH}  collection={COLLECTION_NAME}  count={collection.count()}{RESET}\n")
    except Exception as e:
        print(f"{RED}Failed to open ChromaDB: {e}{RESET}")
        print(f"Make sure {VECTOR_DB_PATH} exists and has been indexed (run scripts/index_metadata.py first).")
        sys.exit(1)

    if args.command == "ids":
        cmd_ids(collection, args.artifact_ids, show_document=args.doc)
    elif args.command == "list":
        cmd_list(collection, args.language, args.artifact_type, args.limit)
    elif args.command == "search":
        cmd_search(collection, args.query, args.limit)
    elif args.command == "gold":
        interactive = not args.no_interactive and sys.stdout.isatty()
        cmd_gold(collection, args.query_id, interactive=interactive)


if __name__ == "__main__":
    main()
