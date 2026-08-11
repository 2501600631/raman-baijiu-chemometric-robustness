                       


from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Restore NumPy aliases removed in newer NumPy releases for compatibility with
# legacy dependencies or historical code that may still reference them.
for _np_alias, _np_actual in {
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "object": object,
}.items():
    if _np_alias not in np.__dict__:
        setattr(np, _np_alias, _np_actual)

import pandas as pd
from sklearn.cross_decomposition import PLSRegression


# Centralized configuration for reproducible MCCV screening.
@dataclass
class Config:
    random_state: int = 2026
    enable_outlier_filter: bool = True
    mccv_n_iter: int = 100
    max_outlier_ratio: float = 0.18
    temporary_calibration_ratio: float = 0.78
    max_pls_components: int = 12


def read_input_tables(
    spectral_file: str,
    phys_file: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    # Load spectral and reference tables from Excel or CSV input files.
    print(f"Reading spectral data: {spectral_file}")
    print(f"Reading reference data: {phys_file}")

    spectral_file = str(spectral_file)
    phys_file = str(phys_file)

    spec = (
        pd.read_excel(spectral_file)
        if spectral_file.lower().endswith((".xlsx", ".xls"))
        else pd.read_csv(spectral_file)
    )
    phys = (
        pd.read_excel(phys_file)
        if phys_file.lower().endswith((".xlsx", ".xls"))
        else pd.read_csv(phys_file)
    )

    # Expected spectral-table layout: first column = spectral axis;
    # remaining columns = samples, with one spectrum per sample column.
    wavelengths = pd.to_numeric(
        spec.iloc[:, 0], errors="coerce"
    ).to_numpy(dtype=float)

    spec_sample_names = spec.columns[1:].astype(str).to_numpy()
    X_all = (
        spec.iloc[:, 1:]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
        .T
    )

    # Expected reference-table layout: first column = analyte names;
    # remaining columns = samples, with analytes stored by row.
    target_names = phys.iloc[:, 0].astype(str).to_numpy()
    target_names = np.array(
        [
            f"Analyte{i + 1}"
            if str(name).strip() == "" or str(name).lower() == "nan"
            else str(name).strip()
            for i, name in enumerate(target_names)
        ],
        dtype=str,
    )

    phys_sample_names = phys.columns[1:].astype(str).to_numpy()
    Y_all = (
        phys.iloc[:, 1:]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
        .T
    )

    # Align both tables strictly by sample name while preserving spectral-table order.
    spec_index = {name: i for i, name in enumerate(spec_sample_names)}
    phys_index = {name: i for i, name in enumerate(phys_sample_names)}

    common = [name for name in spec_sample_names if name in phys_index]
    # A minimum of eight aligned samples is required by the downstream screening logic.
    if len(common) < 8:
        raise ValueError("Fewer than 8 samples can be aligned between the spectral and reference tables; MCCV cannot proceed.")

    X = X_all[[spec_index[n] for n in common], :]
    Y = Y_all[[phys_index[n] for n in common], :]
    sample_names = np.asarray(common, dtype=str)

    print(f"Aligned sample count: {X.shape[0]}")
    print(f"Spectral variable count: {X.shape[1]}")
    print(f"Reference analyte count: {Y.shape[1]}")
    print()

    return wavelengths, sample_names, target_names, X, Y


def mccv_outlier_diagnostics(
    X: np.ndarray,
    y: np.ndarray,
    cfg: Config,
) -> Dict[str, object]:
    
    # Convert inputs to stable NumPy shapes before repeated Monte Carlo splitting.
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()

    n = X.shape[0]
    keep = np.ones(n, dtype=bool)

    # For very small datasets, retain all samples rather than applying unstable screening.
    if not cfg.enable_outlier_filter or n < 15:
        return {
            "keep": keep,
            "candidate_outlier": np.zeros(n, dtype=bool),
            "mean_error": np.full(n, np.nan),
            "prediction_count": np.zeros(n, dtype=int),
            "median_error": np.nan,
            "mad": np.nan,
            "threshold": np.nan,
            "max_outliers_allowed": max(10, int(round(cfg.max_outlier_ratio * n))),
            "filter_applied": False,
            "successful_iterations": 0,
        }

    # pred_err[k, i] stores the absolute prediction error for sample i when
    # that sample appears in the temporary validation subset of MCCV iteration k.
    rng = np.random.default_rng(cfg.random_state)
    pred_err = np.full((cfg.mccv_n_iter, n), np.nan, dtype=float)
    successful_iterations = 0

    # Repeatedly create random calibration/validation splits. Each sample accumulates
    # out-of-split prediction errors across the iterations in which it is validated.
    for k in range(cfg.mccv_n_iter):
        idx = rng.permutation(n)

        # Historical temporary calibration fraction used only for MCCV diagnostics.
        n_train = max(
            8,
            int(round(cfg.temporary_calibration_ratio * n)),
        )

        tr = idx[:n_train]
        te = idx[n_train:]

        if len(te) < 3:
            continue

        try:
            # Limit PLS dimensionality by the configured cap, available variables,
            # and temporary calibration-set size to avoid an invalid component count.
            n_comp = min(
                cfg.max_pls_components,
                X.shape[1],
                len(tr) - 2,
            )
            if n_comp < 1:
                continue

            model = PLSRegression(
                n_components=n_comp,
                scale=True,
            )
            model.fit(X[tr], y[tr])

            # Record only validation-set errors; calibration predictions are not used
            # for sample-level outlier diagnostics.
            pred = model.predict(X[te]).ravel()
            pred_err[k, te] = np.abs(y[te] - pred)
            successful_iterations += 1

        except Exception:
            # A failed temporary submodel is skipped so that isolated numerical issues
            # do not terminate the entire analyte-specific MCCV screening.
            continue

    # Aggregate each sample across all iterations in which it was predicted out-of-split.
    prediction_count = np.sum(np.isfinite(pred_err), axis=0)

    with np.errstate(all="ignore"):
        mean_err = np.nanmean(pred_err, axis=0)

    if np.all(~np.isfinite(mean_err)):
        return {
            "keep": keep,
            "candidate_outlier": np.zeros(n, dtype=bool),
            "mean_error": mean_err,
            "prediction_count": prediction_count,
            "median_error": np.nan,
            "mad": np.nan,
            "threshold": np.nan,
            "max_outliers_allowed": max(10, int(round(cfg.max_outlier_ratio * n))),
            "filter_applied": False,
            "successful_iterations": successful_iterations,
        }

    # Robust threshold: median sample-wise mean absolute error + 3 x MAD.
    med = float(np.nanmedian(mean_err))
    mad = float(np.nanmedian(np.abs(mean_err - med)))
    threshold = med + 3.0 * mad

    candidate_outlier = mean_err > threshold

    # Historical acceptance guard: candidate exclusions are applied only when the
    # candidate count does not exceed the allowed maximum and >=8 samples remain.
    max_outliers = max(
        10,
        int(round(cfg.max_outlier_ratio * n)),
    )

    filter_applied = bool(
        np.sum(~candidate_outlier) >= 8
        and np.sum(candidate_outlier) <= max_outliers
    )

    if filter_applied:
        keep = ~candidate_outlier

    return {
        "keep": keep,
        "candidate_outlier": candidate_outlier,
        "mean_error": mean_err,
        "prediction_count": prediction_count,
        "median_error": med,
        "mad": mad,
        "threshold": threshold,
        "max_outliers_allowed": max_outliers,
        "filter_applied": filter_applied,
        "successful_iterations": successful_iterations,
    }


# Convenience wrapper retained for callers that only need the final keep mask.
def mccv_outlier_keep(
    X: np.ndarray,
    y: np.ndarray,
    cfg: Config,
) -> np.ndarray:
    
    return np.asarray(
        mccv_outlier_diagnostics(X, y, cfg)["keep"],
        dtype=bool,
    )


def run_mccv_screening(
    spectral_file: str,
    reference_file: str,
    output_excel: str,
    output_manifest_csv: str,
    cfg: Config,
):
    
    # Read and align the complete dataset once, then screen each analyte independently.
    _, sample_names, target_names, X, Y = read_input_tables(
        spectral_file,
        reference_file,
    )

    summary_rows: List[Dict[str, object]] = []
    manifest_rows: List[Dict[str, object]] = []
    diagnostic_rows: List[Dict[str, object]] = []

    # Missing reference values are handled analyte by analyte, so each analyte may
    # be screened on a slightly different valid-sample subset.
    for t, target in enumerate(target_names):
        y_all = np.asarray(Y[:, t], dtype=float)
        valid = np.isfinite(y_all)

        X0 = X[valid]
        y0 = y_all[valid]
        sn0 = sample_names[valid]

        print("=" * 80)
        print(f"Analyte: {target}")
        print(f"Valid sample count: {len(y0)}")

        # Skip screening when there are too few valid samples or essentially no
        # response variation; such samples are recorded as not_screened.
        if len(y0) < 8 or np.std(y0) < 1e-12:
            print("Skipping: too few samples or insufficient reference-value variation.")
            for name, value in zip(sn0, y0):
                manifest_rows.append(
                    {
                        "target": str(target),
                        "sampleName": str(name),
                        "referenceValue": float(value),
                        "status": "not_screened",
                        "candidateOutlier": False,
                        "reason": "InsufficientSamplesOrNoResponseVariation",
                    }
                )
            continue

        # Run MCCV diagnostics and retrieve both candidate and final exclusion masks.
        diag = mccv_outlier_diagnostics(X0, y0, cfg)

        keep = np.asarray(diag["keep"], dtype=bool)
        candidate = np.asarray(diag["candidate_outlier"], dtype=bool)
        mean_error = np.asarray(diag["mean_error"], dtype=float)
        prediction_count = np.asarray(diag["prediction_count"], dtype=int)

        n_candidate = int(candidate.sum())
        n_excluded = int((~keep).sum())
        n_retained = int(keep.sum())

        excluded_names = sn0[~keep].astype(str).tolist()
        candidate_names = sn0[candidate].astype(str).tolist()

        print(f"Successful MCCV submodels: {diag['successful_iterations']}/{cfg.mccv_n_iter}")
        print(f"median(mean abs error): {diag['median_error']:.8g}")
        print(f"MAD: {diag['mad']:.8g}")
        print(f"threshold = median + 3*MAD: {diag['threshold']:.8g}")
        print(f"Candidate outlier count: {n_candidate}")
        print(f"Maximum allowed outlier count: {diag['max_outliers_allowed']}")
        print(f"Exclusion applied: {diag['filter_applied']}")
        print(f"Final retained sample count: {n_retained}")
        print(f"Final excluded sample count: {n_excluded}")
        if excluded_names:
            print("Final excluded samples:", ", ".join(excluded_names))
        print()

        # One summary record per analyte captures screening settings and outcomes.
        summary_rows.append(
            {
                "target": str(target),
                "nOriginalValid": int(len(y0)),
                "nRetained": n_retained,
                "nExcluded": n_excluded,
                "nCandidateOutliers": n_candidate,
                "candidateOutlierNames": ",".join(candidate_names),
                "excludedSampleNames": ",".join(excluded_names),
                "medianMeanAbsError": diag["median_error"],
                "MADMeanAbsError": diag["mad"],
                "thresholdMedianPlus3MAD": diag["threshold"],
                "maxOutliersAllowed": int(diag["max_outliers_allowed"]),
                "filterApplied": bool(diag["filter_applied"]),
                "successfulIterations": int(diag["successful_iterations"]),
                "mccvIterations": int(cfg.mccv_n_iter),
                "temporaryCalibrationRatio": float(cfg.temporary_calibration_ratio),
                "maxPLSComponents": int(cfg.max_pls_components),
                "randomState": int(cfg.random_state),
            }
        )

        # Build sample-level inclusion/exclusion provenance and diagnostic records.
        for i, (name, value) in enumerate(zip(sn0, y0)):
            if not keep[i]:
                status = "excluded"
                reason = "MCCV_mean_abs_error_above_median_plus_3MAD"
            else:
                status = "included"
                reason = ""

            if candidate[i] and keep[i]:
                reason = (
                    "CandidateOutlierButFilterNotAppliedBecauseOriginalAcceptanceRuleFailed"
                )

            manifest_rows.append(
                {
                    "target": str(target),
                    "sampleName": str(name),
                    "referenceValue": float(value),
                    "status": status,
                    "candidateOutlier": bool(candidate[i]),
                    "meanAbsolutePredictionError": (
                        float(mean_error[i]) if np.isfinite(mean_error[i]) else np.nan
                    ),
                    "threshold": diag["threshold"],
                    "predictionCount": int(prediction_count[i]),
                    "reason": reason,
                }
            )

            diagnostic_rows.append(
                {
                    "target": str(target),
                    "sampleName": str(name),
                    "referenceValue": float(value),
                    "meanAbsolutePredictionError": (
                        float(mean_error[i]) if np.isfinite(mean_error[i]) else np.nan
                    ),
                    "predictionCount": int(prediction_count[i]),
                    "candidateOutlier": bool(candidate[i]),
                    "finalKeep": bool(keep[i]),
                    "threshold": diag["threshold"],
                    "medianMeanAbsolutePredictionError": diag["median_error"],
                    "MAD": diag["mad"],
                }
            )

    # Materialize accumulated records into tabular outputs.
    summary_df = pd.DataFrame(summary_rows)
    manifest_df = pd.DataFrame(manifest_rows)
    diagnostics_df = pd.DataFrame(diagnostic_rows)

    output_excel_path = Path(output_excel)
    output_excel_path.parent.mkdir(parents=True, exist_ok=True)

    # Excel workbook contains analyte-level summary, sample manifest, and diagnostics.
    with pd.ExcelWriter(output_excel_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        manifest_df.to_excel(writer, sheet_name="Manifest", index=False)
        diagnostics_df.to_excel(writer, sheet_name="Diagnostics", index=False)

    manifest_path = Path(output_manifest_csv)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    # UTF-8 with BOM improves compatibility when the CSV is opened directly in Excel.
    manifest_df.to_csv(
        manifest_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("=" * 80)
    print("MCCV sample screening completed.")
    print(f"Excel output: {output_excel_path.resolve()}")
    print(f"Sample inclusion/exclusion manifest: {manifest_path.resolve()}")

    if not summary_df.empty:
        print("\nSummary:")
        cols = [
            "target",
            "nOriginalValid",
            "nRetained",
            "nExcluded",
            "filterApplied",
            "excludedSampleNames",
        ]
        print(summary_df[cols].to_string(index=False))

    return {
        "summary": summary_df,
        "manifest": manifest_df,
        "diagnostics": diagnostics_df,
    }


# Command-line interface for reproducible standalone execution.
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the standalone historical MCCV analyte-specific sample screening."
    )
    parser.add_argument(
        "--spectral",
        default=str(Path(__file__).resolve().parents[1] / "data" / "raman_sample_mean_spectra_80_samples.xlsx"),
        help="Path to the Raman sample-mean spectra workbook.",
    )
    parser.add_argument(
        "--reference",
        default=str(Path(__file__).resolve().parents[1] / "data" / "gcms_reference_concentrations_8_esters_80_samples.xlsx"),
        help="Path to the GC-MS reference concentration workbook.",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parents[1] / "results" / "MCCV_screening_results.xlsx"),
        help="Excel output file.",
    )
    parser.add_argument(
        "--manifest",
        default=str(Path(__file__).resolve().parents[1] / "results" / "MCCV_sample_inclusion_exclusion_manifest.csv"),
        help="CSV sample inclusion/exclusion manifest.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=2026,
        help="Random seed (default: 2026).",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=100,
        help="Number of successful MCCV submodels requested per analyte (default: 100).",
    )
    parser.add_argument(
        "--max-outlier-ratio",
        type=float,
        default=0.18,
        help="Historical maximum outlier-ratio guard (default: 0.18).",
    )
    return parser.parse_args()


# Script entry point: parse CLI options, construct configuration, and run screening.
if __name__ == "__main__":
    args = parse_args()

    config = Config(
        random_state=args.random_state,
        enable_outlier_filter=True,
        mccv_n_iter=args.n_iter,
        max_outlier_ratio=args.max_outlier_ratio,
        temporary_calibration_ratio=0.78,
        max_pls_components=12,
    )

    run_mccv_screening(
        spectral_file=args.spectral,
        reference_file=args.reference,
        output_excel=args.output,
        output_manifest_csv=args.manifest,
        cfg=config,
    )
