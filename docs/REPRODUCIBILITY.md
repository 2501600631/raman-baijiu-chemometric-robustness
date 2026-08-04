# Reproducibility scope

This repository is designed to support **method transparency, provenance, and partial computational verification** while respecting confidentiality and data-ownership restrictions.

## Publicly included

- Principal/revised modeling implementation.
- Stand-alone MCCV sample-screening implementation.
- Machine-readable analyte-specific inclusion/exclusion manifest.
- Public MCCV screening summary without Raman intensities or reference concentrations.
- Workflow specification and study-defined CRS settings.
- `requirements.txt` and `environment.yml` describing a recommended reproduction environment.
- `capture_environment.py` for recording the exact versions in a local execution environment.
- SHA-256 checksums for the repository snapshot.

## Not publicly included

- Raw Raman spectra.
- GC-MS reference concentrations.
- Confidential sample metadata.
- Complete 806,400-row split-specific result archive.
- Full CRS intermediate/result workbooks and other large derived files.

Because the private input matrices are not public, an unauthorised third party cannot perform a complete end-to-end retraining of every workflow from this repository alone. The public files instead document the analysis design and the fixed screened sample domains. Data access may be requested from the corresponding author, subject to data-owner permission and confidentiality restrictions.

## Historical-analysis note

The manuscript describes an archived analyte-specific, response-informed MCCV screen applied before the shared outer splits. The public manifest records the resulting historical sample domains. The exact historical software version of every package for every pipeline was not preserved; the repository therefore distinguishes the archived analysis provenance from the recommended/current reproduction environment.

Do not reconstruct missing historical artifacts by guesswork. Any later reanalysis should receive a separate version identifier and should be reported as a new analysis rather than silently replacing the archived results.
