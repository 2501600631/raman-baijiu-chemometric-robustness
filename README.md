# Raman-Baijiu chemometric robustness audit

This repository accompanies the manuscript:

**Robustness assessment of Raman chemometric workflows for ester prediction in strong aroma Baijiu**

It provides the experimental input data, principal modeling code, MCCV screening code, fixed analyte-specific sample manifests, workflow specification, and captured software-environment information used to document and reproduce the study.

## Public data

The two experimental input workbooks used for chemometric analysis are included in `data/`:

- `data/raman_sample_mean_spectra_80_samples.xlsx`
  - 994 Raman-shift channels by 80 samples (`S1`-`S80`)
  - first column: `Raman_shift_cm-1`
  - remaining columns: sample-mean Raman intensities
  - each sample spectrum is the mean of three technical replicate spectra
- `data/gcms_reference_concentrations_8_esters_80_samples.xlsx`
  - eight ester analytes by 80 samples (`S1`-`S80`)
  - first column: `Analyte`
  - remaining columns: GC-MS reference concentrations in mg L^-1
  - each reference value is the mean of three determinations

Sample identifiers are aligned across both workbooks. The public analyte names are: Ethyl acetate, Ethyl butyrate, Ethyl hexanoate, Ethyl lactate, Ethyl valerate, Ethyl heptanoate, Ethyl octanoate, and Hexyl hexanoate.

See [`data/README.md`](data/README.md) for a compact data dictionary.

## Repository contents

```text
.
├── README.md
├── CHECKSUMS.sha256
├── CITATION.cff
├── LICENSE
├── DATA_LICENSE.md
├── .gitignore
│
├── data/
│   ├── README.md
│   ├── raman_sample_mean_spectra_80_samples.xlsx
│   └── gcms_reference_concentrations_8_esters_80_samples.xlsx
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

Before the factorial workflow evaluation, an analyte-specific, response-informed Monte Carlo cross-validation (MCCV) screen was applied once to define fixed analyte-specific sample domains. The screening used the complete analyte response vector before the shared outer calibration/prediction splits; this is therefore response-informed pre-split sample selection. Reported downstream performance is conditional on the retained analyte-specific domains.

Historical screening specification:

- 100 successfully fitted MCCV PLSR submodels per analyte;
- approximately 78% temporary calibration samples per submodel;
- PLSRegression with `scale=True` and at most 12 latent variables;
- absolute prediction error recorded only when a sample entered a temporary prediction set;
- sample score = mean absolute prediction error across prediction-set appearances;
- threshold = median sample score + 3 x MAD;
- random state = `2026`;
- exclusion applied only when the historical guard conditions were satisfied.

Retained sample counts:

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

Detailed inclusion/exclusion records are provided in [`manifests/analyte_sample_manifest.csv`](manifests/analyte_sample_manifest.csv).

## Workflow space

The machine-readable design specification is [`docs/workflow_specification.json`](docs/workflow_specification.json). The audited workflow comprised:

- 28 spectral pretreatments;
- 4 response transformations;
- calibration-set proportions of 0.75, 0.80, and 0.85;
- 10 response-stratified outer splits per calibration proportion;
- 30 regression algorithms;
- 10,080 fixed workflow configurations per analyte;
- 100,800 split-specific evaluations per analyte;
- 806,400 split-specific evaluations overall;
- 158 selected Raman variables for each applicable model representation.

Study-defined internal criterion:

```text
Rp^2 >= 0.75
and
|Rc^2 - Rp^2| <= 0.20
```

The composite robustness score (CRS) is a study-specific, post hoc prioritization score. Its component definitions, baseline weights, numerical-failure threshold, and Dirichlet weight-sensitivity settings are recorded in `workflow_specification.json`.

## Reproduction

Install the captured package versions in a compatible Python 3.8 environment:

```bash
python -m pip install -r environment/requirements-lock.txt
```

Validate the repository structure and public datasets:

```bash
python code/validate_public_package.py
```

Re-run the stand-alone historical MCCV screen:

```bash
python code/MCCV_outlier_screening.py
```

Run the principal full workflow search:

```bash
python code/baijiu_bruteforce_full_search.py
```

The full workflow search is computationally intensive. Generated model-search outputs and checkpoints are written under `results/` and are ignored by Git.

## Derived archives

The complete 806,400-row split-specific performance archive and full CRS intermediate/result tables are not included in this repository because they are large derived outputs. The public input data, code, manifests, workflow specification, and environment records are sufficient to inspect the study design and to rerun the released analysis code.

## Software environment

The released scripts were verified in the environment recorded in [`environment/environment_report.json`](environment/environment_report.json). Key captured versions include Python 3.8.3, NumPy 1.23.5, pandas 1.5.3, SciPy 1.10.1, scikit-learn 1.1.3, XGBoost 1.7.6, LightGBM 3.3.5, CatBoost 1.2.3, and PyTorch 2.4.1.

See [`docs/SOFTWARE_ENVIRONMENT.md`](docs/SOFTWARE_ENVIRONMENT.md) for the provenance boundary.

## Integrity verification

SHA-256 hashes for the release snapshot are listed in `CHECKSUMS.sha256`. Recompute them after any intentional modification.

## Citation

Please cite the associated article once final bibliographic information and a DOI are available. A machine-readable citation template is provided in `CITATION.cff`.

## License

Source code is released under the MIT License; see `LICENSE`. The experimental datasets are released under CC BY 4.0; see `DATA_LICENSE.md`. Only publish this repository if all data owners and participating organizations have authorized public release under the stated terms.
