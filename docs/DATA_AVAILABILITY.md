# Data availability

The experimental input data used for chemometric analysis are publicly available in this repository.

Public datasets:

- `data/raman_sample_mean_spectra_80_samples.xlsx`: sample-mean Raman spectra for samples S1-S80 (994 Raman-shift channels x 80 samples).
- `data/gcms_reference_concentrations_8_esters_80_samples.xlsx`: GC-MS reference concentrations for eight ester analytes in samples S1-S80 (8 analytes x 80 samples), reported in mg L^-1.

Sample identifiers are consistent across the two files. An analyte-specific inclusion/exclusion manifest and MCCV screening summary are provided under `manifests/`.

The complete 806,400-row split-specific workflow-performance archive and full CRS intermediate/result tables are large derived outputs and are not included in the GitHub repository. The public source data and released code support independent rerunning of the analysis, subject to computational resources and the software-environment considerations described in `docs/REPRODUCIBILITY.md`.
