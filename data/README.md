# Public experimental input data

This directory contains the two experimental input workbooks used by the released chemometric analysis.

## `raman_sample_mean_spectra_80_samples.xlsx`

- Worksheet: `Raman_spectra`
- Dimensions: 995 rows including the header, 81 columns
- First column: `Raman_shift_cm-1`
- Remaining columns: sample IDs `S1`-`S80`
- Data matrix: 994 Raman-shift channels x 80 samples
- Values: sample-mean Raman intensities in arbitrary units
- Each sample spectrum is the mean of three technical replicate spectra
- Raman-shift range: approximately 200.7894 to 3011.1812 cm^-1
- The Raman-shift grid is shared by all 80 samples and is not perfectly equidistant

## `gcms_reference_concentrations_8_esters_80_samples.xlsx`

- Worksheet: `GCMS_reference`
- Dimensions: 9 rows including the header, 81 columns
- First column: `Analyte`
- Remaining columns: sample IDs `S1`-`S80`
- Values: GC-MS reference concentrations in mg L^-1
- Each reference concentration is the mean of three determinations
- Analytes, in row order:
  1. Ethyl acetate
  2. Ethyl butyrate
  3. Ethyl hexanoate
  4. Ethyl lactate
  5. Ethyl valerate
  6. Ethyl heptanoate
  7. Ethyl octanoate
  8. Hexyl hexanoate

## Alignment

The sample identifiers `S1`-`S80` are consistent across both workbooks. The released code aligns samples by these identifiers rather than by column position alone.

## Scope

These files contain the sample-level inputs used for chemometric analysis. They do not include the three individual Raman technical-replicate spectra or the three individual GC-MS replicate determinations because the manuscript analysis used the per-sample means.
