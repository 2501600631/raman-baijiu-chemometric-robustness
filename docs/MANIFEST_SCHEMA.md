# Analyte-specific sample manifest

`manifests/analyte_sample_manifest.csv` records the fixed analyte-specific sample domains used after the archived MCCV screen.

Fields:

- `analyte`: English analyte name.
- `abbreviation`: manuscript abbreviation.
- `target_label_in_reference_file`: analyte label in the public GC-MS reference workbook.
- `sample_id`: anonymized sample identifier (`S1`-`S80`).
- `status`: `included` or `excluded` for that analyte.
- `screening_basis`: identifies the archived response-informed analyte-specific MCCV screen.
- `exclusion_reason`: rule-level reason for excluded samples.

Historical screening settings: 100 successfully fitted MCCV PLSR submodels, approximately 78% temporary calibration fraction, maximum 12 PLS components, threshold `mean absolute prediction error > median + 3 MAD`, random state 2026, and the historical maximum-outlier-ratio guard.

The manifest documents the historical screened domains. It is not an independently validated quality-control exclusion standard and should not be interpreted as one.
