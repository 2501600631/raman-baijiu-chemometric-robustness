# Raman–Baijiu chemometric robustness audit

This repository contains the public code and reproducibility/provenance materials accompanying the manuscript:

**“A robustness audit of chemometric workflows for Raman fingerprint regression of eight esters in strong-aroma Baijiu.”**

## Scope

The study evaluated a fixed Raman–chemometric workflow space for eight ester analytes in strong-aroma Baijiu. This repository is intended to support **method transparency, provenance checking, inspection of the historical analyte-specific sample domains, and partial computational verification**.

The repository does **not** publicly distribute the underlying experimental Raman or GC–MS sample matrices.

## Data availability

The raw experimental data are not publicly available because of confidentiality and data-ownership restrictions. This includes:

- the 80 × 994 Raman spectral matrix;
- the 80 × 8 GC–MS reference-concentration matrix;
- confidential sample-level experimental metadata.

Access to the underlying experimental data may be requested from the **corresponding author** on reasonable request, subject to permission from the data owner and applicable confidentiality restrictions.

The complete 806,400 split-specific evaluation records and the full CRS intermediate/result tables are retained by the authors but are not included in this GitHub repository.

See [`docs/DATA_AVAILABILITY.md`](docs/DATA_AVAILABILITY.md) and [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for details.

## Repository contents

```text
.
├── README.md
├── CHECKSUMS.sha256
├── .gitignore
│
├── code/
│   ├── MCCV_outlier_screening.py
│   ├── baijiu_bruteforce_full_search.py
│   └── validate_public_package.py
│
├── manifests/
│   ├── analyte_sample_manifest.csv
│   └── MCCV_screening_summary_public.csv
│
├── environment/
│   ├── environment_report.json
│   └── requirements-lock.txt
│
└── docs/
    ├── DATA_AVAILABILITY.md
    ├── MANIFEST_SCHEMA.md
    ├── REPRODUCIBILITY.md
    ├── SOFTWARE_ENVIRONMENT.md
    ├── SUGGESTED_STATEMENTS.txt
    └── workflow_specification.json
```

## Historical MCCV sample screening

Before the historical factorial workflow evaluation, an analyte-specific, response-informed Monte Carlo cross-validation (MCCV) screen was applied to define fixed analyte-specific sample domains.

The public machine-readable manifest records the inclusion/exclusion status of anonymized sample identifiers **S1–S80** for each analyte. It does not contain Raman intensities or GC–MS concentration values.

The historical screening specification was:

- 100 MCCV submodels per analyte;
- approximately 78% temporary calibration samples per submodel;
- PLS regression with up to 12 latent variables;
- sample-level mean absolute prediction error calculated when a sample entered a temporary prediction set;
- screening threshold: `median(mean absolute prediction error) + 3 × MAD`;
- random state: `2026`.

The retained sample counts were:

| Analyte | Abbreviation | Retained n |
|---|---:|---:|
| Ethyl acetate | EA | 72 |
| Ethyl butyrate | EB | 66 |
| Ethyl hexanoate | EH | 71 |
| Ethyl lactate | EL | 69 |
| Ethyl valerate | EV | 74 |
| Ethyl heptanoate | EHp | 74 |
| Ethyl octanoate | EO | 74 |
| Hexyl hexanoate | HH | 76 |

The detailed anonymized inclusion/exclusion records are provided in [`manifests/analyte_sample_manifest.csv`](manifests/analyte_sample_manifest.csv).

## Workflow specification

A machine-readable description of the study design is provided in [`docs/workflow_specification.json`](docs/workflow_specification.json).

The historical workflow space comprised:

- 28 spectral pretreatments;
- 4 response transformations;
- 3 calibration-set proportions: 0.75, 0.80, and 0.85;
- 10 response-stratified outer splits per calibration proportion;
- 30 regression algorithms;
- 10,080 fixed workflow configurations per analyte;
- 100,800 split-specific evaluations per analyte;
- 806,400 split-specific evaluations overall;
- 158 selected Raman variables for each applicable model representation.

The study-defined internal criterion was:

```text
Rp² >= 0.75
and
|Rc² - Rp²| <= 0.20
```

The composite robustness score (CRS) was a study-specific, post hoc prioritization score. Its component definitions, weights, numerical-failure threshold, and Dirichlet weight-sensitivity settings are recorded in `workflow_specification.json`.

## Software environment

The released public scripts were verified in the captured environment recorded in [`environment/environment_report.json`](environment/environment_report.json).

Key versions were:

```text
Python            3.8.3
NumPy             1.23.5
pandas            1.5.3
SciPy             1.10.1
scikit-learn      1.1.3
openpyxl          3.1.5
joblib            1.1.0
XGBoost           1.7.6
LightGBM          3.3.5
CatBoost          1.2.3
PyTorch           2.4.1
```

The complete captured package list used for this public verification environment is provided in [`environment/requirements-lock.txt`](environment/requirements-lock.txt).

This captured environment documents the environment in which the released code was verified. It should **not** be interpreted as proof that the exact package version of every historical archived model fit was preserved.

For additional notes, see [`docs/SOFTWARE_ENVIRONMENT.md`](docs/SOFTWARE_ENVIRONMENT.md).

## Using the public files

To inspect the fixed historical sample domains:

```text
manifests/analyte_sample_manifest.csv
```

To inspect the public MCCV screening summary:

```text
manifests/MCCV_screening_summary_public.csv
```

To inspect the study configuration and CRS settings:

```text
docs/workflow_specification.json
```

To install the captured package versions in a compatible Python 3.8 environment:

```bash
python -m pip install -r environment/requirements-lock.txt
```

To validate the public package:

```bash
python code/validate_public_package.py
```

The principal modeling implementation included in this repository is:

```text
code/baijiu_bruteforce_full_search.py
```

## Reproducibility boundary

Because the experimental Raman and GC–MS matrices are not publicly available, this repository alone cannot support complete end-to-end retraining of all 806,400 historical workflow evaluations.

The public materials are intended to document:

1. the historical analyte-specific sample domains;
2. the MCCV screening implementation;
3. the principal modeling implementation;
4. the fixed workflow design and CRS specification;
5. the captured software environment used to verify the released public scripts;
6. the integrity of the public repository snapshot.

Researchers requiring access to the underlying experimental data should contact the corresponding author, subject to data-owner permission and applicable confidentiality restrictions.

## Integrity verification

SHA-256 hashes for the public repository snapshot are listed in:

```text
CHECKSUMS.sha256
```

These hashes can be used to verify that the public files have not changed relative to the recorded snapshot.

## Citation

If this repository is used in scientific work, please cite the associated article once the final bibliographic information and DOI are available.

## License

No license is asserted unless a separate `LICENSE` file is added to the repository. Users should not assume rights beyond those explicitly granted by the repository owner or the associated publication.
