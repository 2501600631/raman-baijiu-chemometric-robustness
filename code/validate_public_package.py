#!/usr/bin/env python3
"""Validate the small public GitHub reproducibility package without private data."""
from __future__ import annotations
import csv
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 25 * 1024 * 1024
EXPECTED = {
    "EA": 72, "EB": 66, "EH": 71, "EL": 69,
    "EV": 74, "EHp": 74, "EO": 74, "HH": 76,
}

# GitHub-size check requested for this public package.
oversized = [(p.relative_to(ROOT), p.stat().st_size) for p in ROOT.rglob("*") if p.is_file() and p.stat().st_size >= MAX_BYTES]
if oversized:
    raise SystemExit(f"Files >=25 MiB detected: {oversized}")

manifest = ROOT / "manifests" / "analyte_sample_manifest.csv"
rows = list(csv.DictReader(manifest.open(encoding="utf-8-sig", newline="")))
if len(rows) != 640:
    raise SystemExit(f"Expected 640 manifest rows, found {len(rows)}")

included = Counter(r["abbreviation"] for r in rows if r["status"] == "included")
excluded = Counter(r["abbreviation"] for r in rows if r["status"] == "excluded")
for abbr, expected_n in EXPECTED.items():
    if included[abbr] != expected_n:
        raise SystemExit(f"{abbr}: expected {expected_n} included, found {included[abbr]}")
    if included[abbr] + excluded[abbr] != 80:
        raise SystemExit(f"{abbr}: included+excluded != 80")

# Public manifest must not contain raw measurement fields.
for forbidden in ("referenceValue", "concentration", "Raman", "intensity"):
    if forbidden in rows[0]:
        raise SystemExit(f"Forbidden experimental-data column in public manifest: {forbidden}")

print("Public package validation passed.")
print("Manifest rows: 640")
print("Retained counts:", dict(included))
print("All public files are <25 MiB.")
