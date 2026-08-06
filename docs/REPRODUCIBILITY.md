# Reproducibility scope

This repository is designed to support method transparency, provenance inspection, and end-to-end computational rerunning from the released sample-mean experimental inputs.

## Publicly included

- Sample-mean Raman spectra for S1-S80.
- GC-MS reference concentrations for the eight target esters in S1-S80.
- Principal modeling implementation.
- Stand-alone MCCV sample-screening implementation.
- Machine-readable analyte-specific inclusion/exclusion manifest.
- MCCV screening summary.
- Machine-readable workflow specification and study-defined CRS settings.
- Captured software-environment records.
- SHA-256 checksums for the repository snapshot.

## Not included

- Individual Raman technical-replicate spectra before per-sample averaging.
- Individual GC-MS replicate determinations before per-sample averaging.
- Complete 806,400-row split-specific derived performance archive.
- Full CRS intermediate/result workbooks and other large generated outputs.

The released input workbooks are the same sample-level matrices used by the chemometric analysis described in the manuscript. A full rerun remains computationally intensive and may be sensitive to operating-system, library, hardware, and stochastic implementation details even when deterministic seeds are fixed.

## Historical-analysis note

The manuscript describes an archived analyte-specific, response-informed MCCV screen applied before the shared outer splits. The public manifest records the resulting historical sample domains. This screening is intentionally documented rather than hidden: downstream performance is conditional on the retained analyte-specific domains.

The captured environment documents the environment in which the released code was verified. It does not establish that every historical archived model fit was produced with exactly the same package build or hardware. Any later reanalysis should use a separate version identifier rather than silently replacing the archived study results.
