# Release 1.0.1

Prepared on 2026-10-12 from public commit 4ec6cfeee59a585eb3394557deb5c8d9e2cf0ac6.

- Removed comments and docstrings from all Python scripts. Model and MCCV execution structures were compared before and after cleanup, ignoring removed docstrings, and were unchanged.
- Removed the incorrect code comment that described 806,400 evaluations per analyte. The design is 28 x 4 x 3 x 10 x 30 = 100,800 evaluations per analyte and 806,400 across eight analytes.
- Preserved all experimental cells, sample identifiers, sample manifests, historical screening summaries, model parameters, scientific workflow settings, and numerical operations. English font aliases and style names replace Chinese workbook metadata; all workbook parts other than styles.xml remain byte-identical.
- Added release-integrity validation covering all SHA-256 entries, the complete file inventory, paths, text files, workbook XML, Python syntax, and the absence of code comments and docstrings.
- Regenerated the checksum manifest after all file changes.

Validation of this update covers source structure, package integrity, and experimental input consistency. The 806,400 historical model evaluations were not rerun for a comment-removal update.

The modeling code locks the script hash for checkpoint provenance. Checkpoints created with an earlier script snapshot may require a new results directory. The historical software-environment record is retained as recorded; it is not a record of a new full model run.
