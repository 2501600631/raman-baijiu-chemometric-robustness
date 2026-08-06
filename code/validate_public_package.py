#!/usr/bin/env python3
"""Validate the public GitHub release package and its experimental inputs."""
from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 25 * 1024 * 1024
EXPECTED_RETAINED = {
    "EA": 72,
    "EB": 66,
    "EH": 71,
    "EL": 69,
    "EV": 74,
    "EHp": 74,
    "EO": 74,
    "HH": 76,
}
EXPECTED_ANALYTES = [
    "Ethyl acetate",
    "Ethyl butyrate",
    "Ethyl hexanoate",
    "Ethyl lactate",
    "Ethyl valerate",
    "Ethyl heptanoate",
    "Ethyl octanoate",
    "Hexyl hexanoate",
]
EXPECTED_SAMPLES = [f"S{i}" for i in range(1, 81)]
REQUIRED_FILES = [
    "README.md",
    "LICENSE",
    "DATA_LICENSE.md",
    "CITATION.cff",
    "data/README.md",
    "data/raman_sample_mean_spectra_80_samples.xlsx",
    "data/gcms_reference_concentrations_8_esters_80_samples.xlsx",
    "code/MCCV_outlier_screening.py",
    "code/baijiu_bruteforce_full_search.py",
    "manifests/analyte_sample_manifest.csv",
    "manifests/MCCV_screening_summary_public.csv",
    "docs/workflow_specification.json",
    "environment/environment_report.json",
    "environment/requirements-lock.txt",
]


def fail(message: str) -> None:
    raise SystemExit(message)


for rel in REQUIRED_FILES:
    if not (ROOT / rel).is_file():
        fail(f"Required file is missing: {rel}")

oversized = [
    (p.relative_to(ROOT), p.stat().st_size)
    for p in ROOT.rglob("*")
    if p.is_file() and p.stat().st_size >= MAX_BYTES
]
if oversized:
    fail(f"Files >=25 MiB detected: {oversized}")

# Validate the English Raman input workbook.
raman_path = ROOT / "data" / "raman_sample_mean_spectra_80_samples.xlsx"
raman = pd.read_excel(raman_path)
if raman.shape != (994, 81):
    fail(f"Unexpected Raman table shape: {raman.shape}; expected (994, 81)")
if str(raman.columns[0]) != "Raman_shift_cm-1":
    fail(f"Unexpected Raman first-column label: {raman.columns[0]!r}")
if list(map(str, raman.columns[1:])) != EXPECTED_SAMPLES:
    fail("Raman sample columns are not exactly S1-S80 in order.")
if raman.iloc[:, 0].isna().any() or raman.iloc[:, 1:].isna().any().any():
    fail("Missing values detected in the public Raman workbook.")

# Validate the English GC-MS reference workbook.
gcms_path = ROOT / "data" / "gcms_reference_concentrations_8_esters_80_samples.xlsx"
gcms = pd.read_excel(gcms_path)
if gcms.shape != (8, 81):
    fail(f"Unexpected GC-MS table shape: {gcms.shape}; expected (8, 81)")
if str(gcms.columns[0]) != "Analyte":
    fail(f"Unexpected GC-MS first-column label: {gcms.columns[0]!r}")
if list(map(str, gcms.columns[1:])) != EXPECTED_SAMPLES:
    fail("GC-MS sample columns are not exactly S1-S80 in order.")
if gcms.iloc[:, 0].astype(str).tolist() != EXPECTED_ANALYTES:
    fail("GC-MS analyte names or analyte order do not match the manuscript release.")
if gcms.iloc[:, 1:].isna().any().any():
    fail("Missing values detected in the public GC-MS reference workbook.")

# Validate the fixed analyte-specific historical manifest.
manifest_path = ROOT / "manifests" / "analyte_sample_manifest.csv"
with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
    rows = list(csv.DictReader(handle))
if len(rows) != 640:
    fail(f"Expected 640 manifest rows, found {len(rows)}")

included = Counter(r["abbreviation"] for r in rows if r["status"] == "included")
excluded = Counter(r["abbreviation"] for r in rows if r["status"] == "excluded")
for abbr, expected_n in EXPECTED_RETAINED.items():
    if included[abbr] != expected_n:
        fail(f"{abbr}: expected {expected_n} included, found {included[abbr]}")
    if included[abbr] + excluded[abbr] != 80:
        fail(f"{abbr}: included + excluded does not equal 80")

for row in rows:
    if row["target_label_in_reference_file"] != row["analyte"]:
        fail("Manifest target labels are not aligned to the public English reference workbook.")

# Validate the public workflow specification.
spec = json.loads((ROOT / "docs" / "workflow_specification.json").read_text(encoding="utf-8"))
if spec.get("original_sample_count") != 80:
    fail("workflow_specification.json does not report 80 original samples.")
if spec.get("factorial_workflow", {}).get("split_specific_evaluations_total") != 806400:
    fail("workflow_specification.json does not report 806,400 total split-specific evaluations.")

# English-only text check for repository-authored text files.
han = re.compile(r"[\u4e00-\u9fff]")
text_suffixes = {".md", ".txt", ".py", ".json", ".csv", ".cff", ".gitignore", ""}
for path in ROOT.rglob("*"):
    if not path.is_file() or path.name == "CHECKSUMS.sha256":
        continue
    if path.suffix.lower() not in text_suffixes and path.name not in {".gitignore", "LICENSE"}:
        continue
    try:
        content = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        continue
    if han.search(content):
        fail(f"Chinese characters detected in repository text file: {path.relative_to(ROOT)}")

print("Public package validation passed.")
print("Raman input: 994 channels x 80 samples")
print("GC-MS input: 8 analytes x 80 samples")
print("Manifest rows: 640")
print("Retained counts:", dict(included))
print("All repository-authored text files are English-only.")
print("All public files are <25 MiB.")
