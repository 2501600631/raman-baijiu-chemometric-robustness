# Public analyte-specific sample manifest

`analyte_sample_manifest.csv` is a machine-readable record of the fixed analyte-specific sample domains used after the archived MCCV screen.

It intentionally contains **no Raman intensities, reference concentrations, production-year metadata, or other experimental measurements**.

Fields:

- `analyte`: English analyte name.
- `abbreviation`: manuscript abbreviation.
- `target_label_in_reference_file`: analyte label used in the private reference workbook.
- `sample_id`: anonymized identifier S1-S80.
- `status`: `included` or `excluded` for that analyte.
- `screening_basis`: identifies the archived response-informed analyte-specific MCCV screen.
- `exclusion_reason`: rule-level reason for excluded samples.

Screening settings: 100 MCCV PLSR submodels, approximately 78% temporary calibration fraction, maximum 12 PLS components, threshold `mean absolute prediction error > median + 3 MAD`, random state 2026.

The manifest is metadata for reproducing the historical screened domains; it is not an independently justified quality-control exclusion list and should not be interpreted as such.
