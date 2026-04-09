"""Sanity check for the reviewed gold dataset."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.gold_dataset import GoldDataset

ds = GoldDataset.load("tests/fixtures/gold_dataset_review.json")
print(f"{len(ds)} queries")
for q in ds:
    total = sum(len(v) for v in q.relevant_artifacts.values())
    print(f"  {q.id}: {total} relevant artifacts")
