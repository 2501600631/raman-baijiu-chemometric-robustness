# Raman–Baijiu chemometric robustness audit

Repository companion materials for the manuscript **“A robustness audit of chemometric workflows for Raman fingerprint regression of eight esters in strong-aroma Baijiu.”**

## Scope

The repository provides analysis code and small, non-sensitive reproducibility metadata. Raw experimental sample data are not publicly distributed because of confidentiality and data-ownership restrictions. Access may be requested from the corresponding author on reasonable request, subject to data-owner permission and applicable restrictions.

## Repository layout

```text
code/
  baijiu_bruteforce_full_search_PAPER_ALIGNED_FINAL_CORRECTED.py
  MCCV_outlier_screening.py
manifests/
  analyte_sample_manifest.csv
  MCCV_screening_summary_public.csv
environment/
  requirements.txt
  environment.yml
  capture_environment.py
docs/
  DATA_AVAILABILITY.md
  MANIFEST_SCHEMA.md
  REPRODUCIBILITY.md
  SOFTWARE_ENVIRONMENT.md
  workflow_specification.json
  SUGGESTED_STATEMENTS.txt
CHECKSUMS.sha256
.gitignore
```

## Historical sample screening

The archived analysis used an analyte-specific response-informed Monte Carlo cross-validation screen before the shared outer calibration/prediction splits. The public manifest records the resulting inclusion/exclusion status for anonymized sample IDs S1-S80. No Raman intensity or GC-MS concentration is contained in the manifest.

Retained sample counts are EA 72, EB 66, EH 71, EL 69, EV 74, EHp 74, EO 74, and HH 76.

## Software

A recommended Python 3.10 environment is given in `environment/environment.yml` and `environment/requirements.txt`. Because exact historical versions were not preserved for every archived pipeline, these files should be described as a recommended/current reproduction environment rather than the exact historical environment.

After validating the code locally, run:

```bash
python environment/capture_environment.py
```

and commit the generated `environment_report.json` and `requirements-lock.txt`.

## Data and large result archives

The following are intentionally not stored in this public GitHub repository:

- raw 80 × 994 Raman spectral matrix;
- 80 × 8 GC-MS reference-concentration matrix;
- confidential sample-level experimental metadata;
- complete 806,400-row split-specific result archive;
- full CRS intermediate/result tables.

See `docs/DATA_AVAILABILITY.md` and `docs/REPRODUCIBILITY.md`.

## Reproducibility boundary

Without authorized access to the private experimental matrices, the public repository does not support complete end-to-end retraining of every model. It supports transparent inspection of the implemented methods, software requirements, screening domains, and study configuration.
