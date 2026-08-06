from __future__ import annotations

import os
import sys
import math
import time
import json
import hashlib
import pickle
import warnings
import threading
import platform
from importlib import metadata as importlib_metadata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np


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
from joblib import Parallel, delayed
from scipy.signal import savgol_filter
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
)


try:
    from sklearn.ensemble import HistGradientBoostingRegressor
    HISTGBR_AVAILABLE = True
except Exception:
    try:
        from sklearn.experimental import enable_hist_gradient_boosting
        from sklearn.ensemble import HistGradientBoostingRegressor
        HISTGBR_AVAILABLE = True
    except Exception:
        HistGradientBoostingRegressor = None
        HISTGBR_AVAILABLE = False


try:
    from sklearn.ensemble import StackingRegressor
    STACKING_REGRESSOR_AVAILABLE = True
except Exception:
    StackingRegressor = None
    STACKING_REGRESSOR_AVAILABLE = False
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, RidgeCV, Lasso, ElasticNet, BayesianRidge, HuberRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


warnings.filterwarnings("default", category=ConvergenceWarning)
warnings.filterwarnings("default", category=RuntimeWarning)


try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except Exception:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except Exception:
    CATBOOST_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


@dataclass
class Config:
    random_state: int = 2026
    pipeline_version: str = "RAMAN_BAIJIU_ROBUSTNESS_PUBLIC_V1"


    full_search: bool = True


    preprocess_methods: Tuple[str, ...] = (
        "RAW", "CENTER", "STANDARDIZE", "MINMAX", "ROBUST",
        "VECTOR_NORM", "AREA_NORM", "SG", "SG1", "SG2", "MSC", "SNV",
        "DETREND", "BASELINE_POLY2", "BASELINE_POLY3", "SNV-SG", "MSC-SG",
        "SG-SNV", "SG-MSC", "DE1", "DE2", "DE1-SNV", "DE1-MSC",
        "DE2-SNV", "DE2-MSC", "BASELINE_POLY2-SNV",
        "BASELINE_POLY2-MSC", "BASELINE_POLY2-SG",
    )


    target_transforms: Tuple[str, ...] = ("raw", "log1p", "sqrt", "standard")


    train_ratio_list: Tuple[float, ...] = (0.75, 0.80, 0.85)
    n_repeated_split: int = 10


    algorithms: Tuple[str, ...] = (
        "PLS", "Ridge", "LASSO", "ElasticNet", "BayesianRidge", "Huber",
        "PCR", "SVR-RBF", "SVR-Linear", "KRR-RBF", "RBF-LS-SVM",
        "RandomForest", "ExtraTrees", "GradientBoosting", "HistGBR",
        "AdaBoost", "KNN", "ANN-MLP", "ELM", "GPR-SE", "GPR-Matern32",
        "PLS-LSSVM", "WeightedStacking", "SuperStacking",
    )
    use_xgboost: bool = True
    use_lightgbm: bool = True
    use_catboost: bool = True


    enable_deep_learning: bool = True
    deep_algorithms: Tuple[str, ...] = ("TorchMLP", "Torch1DCNN", "TorchCNNGRU")
    deep_epochs: int = 140
    deep_patience: int = 18
    deep_batch_size: int = 16
    deep_lr: float = 1e-3
    deep_weight_decay: float = 5e-3
    deep_validation_fraction: float = 0.20
    use_gpu_for_torch: bool = True


    enable_feature_selection: bool = True
    feature_selection_mode: str = "fixed_count_nested"
    fixed_feature_count: int = 158
    feature_count_candidates: Tuple[int, ...] = (158,)
    feature_selection_inner_folds: int = 4
    feature_selection_inner_repeats: int = 1
    feature_selection_one_se: bool = False
    feature_ranking_pls_components: int = 8
    feature_selection_min_vars: int = 12


    enable_outlier_filter: bool = False
    outlier_filter_scope: str = "disabled"
    mccv_n_iter: int = 100
    max_outlier_ratio: float = 0.10


    max_pls_components: int = 12
    max_pcr_components: int = 12
    inner_cv_folds: int = 5
    inner_cv_repeats: int = 2
    use_one_standard_error_rule: bool = True


    strict_leakage_guard: bool = True
    shared_outer_splits: bool = True
    strict_stacking_nested_preprocess: bool = True
    outer_split_method: str = "response_stratified_random"
    require_exact_30_algorithms: bool = True
    require_complete_factorial: bool = True


    search_level: str = "balanced"


    n_jobs: int = 6
    parallel_backend: str = "threading"


    show_realtime_log: bool = True
    show_failed_reason: bool = True
    show_detail_each_target: bool = True
    show_top_n: int = 50
    save_prediction_details: bool = True


    output_excel: str = "raman_baijiu_full_search_results.xlsx"
    output_model_dir: str = "results/baijiu_bruteforce_models"


    enable_resume: bool = True
    checkpoint_file: str = "baijiu_bruteforce_checkpoint.pkl"
    save_checkpoint_every_target: bool = True


    predeclared_exclusion_file: Optional[str] = None
    require_reason_for_missing_reference: bool = False


    output_compatibility_mode: str = "legacy_four_sheet"
    save_auxiliary_audit_csv: bool = True


@dataclass
class ModelResult:
    target: str
    model_name: str
    preprocess: str
    y_transform: str
    train_ratio: float
    repeat_id: int
    split_id: int
    n_original_samples: int
    n_samples: int
    n_presearch_outliers: int
    n_calibration_raw: int
    n_calibration_used: int
    n_prediction: int
    n_calibration_outliers: int
    n_features: int
    n_effective_variables: int
    rc2: float
    rp2: float
    rmsec: float
    rmsep: float
    mae: float
    rpd: float
    y_train: np.ndarray
    y_test: np.ndarray
    pred_train: np.ndarray
    pred_test: np.ndarray
    train_sample_names: List[str]
    test_sample_names: List[str]
    presearch_outlier_names: List[str]
    calibration_outlier_names: List[str]
    selected_variable_indices: np.ndarray
    selected_wavenumbers: np.ndarray
    elapsed_sec: float
    outer_split_id: int = 0
    warning_count: int = 0
    warning_messages: List[str] = field(default_factory=list)
    selection_scope: str = "post_selection_internal_holdout_upper_envelope"
    error: str = ""

    def metric_row(self) -> Dict[str, Any]:
        return {
            "target": self.target,
            "modelName": self.model_name,
            "preprocess": self.preprocess,
            "yTransform": self.y_transform,
            "trainRatio": self.train_ratio,
            "repeatId": self.repeat_id,
            "splitId": self.split_id,
            "Rc2": self.rc2,
            "Rp2": self.rp2,
            "RMSEC": self.rmsec,
            "RMSEP": self.rmsep,
            "MAE": self.mae,
            "RPD": self.rpd,
            "nOriginalSamples": self.n_original_samples,
            "nSamples": self.n_samples,
            "nPresearchOutliers": self.n_presearch_outliers,
            "nCalibrationRaw": self.n_calibration_raw,
            "nCalibrationUsed": self.n_calibration_used,
            "nPrediction": self.n_prediction,
            "nCalibrationOutliers": self.n_calibration_outliers,
            "calibrationOutlierNames": ",".join(self.calibration_outlier_names),
            "nFeatures": self.n_features,
            "nEffectiveVariables": self.n_effective_variables,
            "selectedVariableIndices": ",".join(map(str, self.selected_variable_indices.tolist())),
            "selectedRamanShifts_cm-1": ",".join(f"{x:.6f}" for x in self.selected_wavenumbers.tolist()),
            "elapsedSec": self.elapsed_sec,
        }


@dataclass
class SplitSpec:

    split_id: int
    train_ratio: float
    repeat_id: int
    train_idx_raw: np.ndarray
    train_idx: np.ndarray
    test_idx: np.ndarray
    n_calibration_outliers: int = 0

def stable_seed(base_seed: int, *parts: Any) -> int:

    payload = "|".join([str(base_seed)] + [str(x) for x in parts]).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def complete_oof_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:

    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.shape != y_pred.shape or not np.all(np.isfinite(y_pred)):
        return float("inf")
    return safe_rmse(y_true, y_pred)


def _sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:

    p = Path(path)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"File required for provenance lock does not exist: {p}")
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _sha256_text_sequence(values: Iterable[Any]) -> str:
    payload = json.dumps(
        [str(x) for x in values],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_numeric_array(values: np.ndarray) -> str:

    arr = np.asarray(values, dtype="<f8", order="C")
    h = hashlib.sha256()
    h.update(str(arr.shape).encode("ascii"))
    h.update(arr.tobytes(order="C"))
    return h.hexdigest()


def _installed_version(distribution_name: str) -> str:
    try:
        return importlib_metadata.version(distribution_name)
    except Exception:
        return "not-installed"


def _script_identity() -> Dict[str, Any]:
    try:
        script_path = Path(__file__).resolve()
        return {
            "fileName": script_path.name,
            "sha256": _sha256_file(str(script_path)),
        }
    except Exception as exc:
        raise RuntimeError(
            f"Unable to lock the identity of the current Python script; checkpoint execution cannot continue: {exc}"
        ) from exc


def _environment_lock(cfg: Config) -> Dict[str, Any]:

    cuda_available = bool(TORCH_AVAILABLE and torch.cuda.is_available())
    torch_device_mode = "cuda" if (cfg.use_gpu_for_torch and cuda_available) else "cpu"
    cuda_device_name = None
    if torch_device_mode == "cuda":
        try:
            cuda_device_name = str(torch.cuda.get_device_name(0))
        except Exception:
            cuda_device_name = "unresolved"

    return {
        "pythonVersion": platform.python_version(),
        "pythonImplementation": platform.python_implementation(),
        "operatingSystem": platform.system(),
        "machine": platform.machine(),
        "numpyVersion": str(np.__version__),
        "pandasVersion": str(pd.__version__),
        "scipyVersion": _installed_version("scipy"),
        "scikitLearnVersion": _installed_version("scikit-learn"),
        "joblibVersion": _installed_version("joblib"),
        "openpyxlVersion": _installed_version("openpyxl"),
        "xgboostVersion": _installed_version("xgboost"),
        "lightgbmVersion": _installed_version("lightgbm"),
        "catboostVersion": _installed_version("catboost"),
        "torchVersion": _installed_version("torch"),
        "torchConfiguredUseGPU": bool(cfg.use_gpu_for_torch),
        "torchCUDAAvailable": cuda_available,
        "torchDeviceMode": torch_device_mode,
        "cudaDeviceName": cuda_device_name,
        "parallelBackend": str(cfg.parallel_backend),
        "nJobs": int(cfg.n_jobs),
    }


def build_provenance_context(
    spectral_file: str,
    phys_file: str,
    wavelengths: np.ndarray,
    sample_names: np.ndarray,
    target_names: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    cfg: Config,
) -> Dict[str, Any]:

    spectral_path = Path(spectral_file)
    phys_path = Path(phys_file)

    wavelength_hash = _sha256_numeric_array(wavelengths)
    sample_hash = _sha256_text_sequence(sample_names)
    target_hash = _sha256_text_sequence(target_names)
    spectral_matrix_hash = _sha256_numeric_array(X)
    reference_matrix_hash = _sha256_numeric_array(Y)

    paired = hashlib.sha256()
    for value in (
        wavelength_hash,
        sample_hash,
        target_hash,
        spectral_matrix_hash,
        reference_matrix_hash,
    ):
        paired.update(value.encode("ascii"))

    return {
        "signatureSchemaVersion": 2,
        "inputFiles": {
            "spectralFileName": spectral_path.name,
            "spectralFileSHA256": _sha256_file(str(spectral_path)),
            "referenceFileName": phys_path.name,
            "referenceFileSHA256": _sha256_file(str(phys_path)),
        },
        "alignedData": {
            "nSamples": int(X.shape[0]),
            "nRamanVariables": int(X.shape[1]),
            "nTargets": int(Y.shape[1]),
            "sampleIdsSHA256": sample_hash,
            "targetNamesSHA256": target_hash,
            "wavelengthAxisSHA256": wavelength_hash,
            "spectralMatrixSHA256": spectral_matrix_hash,
            "referenceMatrixSHA256": reference_matrix_hash,
            "pairedDatasetSHA256": paired.hexdigest(),
        },
        "script": _script_identity(),
        "environment": _environment_lock(cfg),
    }


def scientific_config_payload(
    cfg: Config,
    algorithms: Optional[List[str]] = None,
    provenance: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:

    if provenance is None:
        raise RuntimeError(
            "Run signature is missing data/code/environment provenance; "
            "the release checkpoint cannot run without locked provenance."
        )

    return {
        "pipelineVersion": cfg.pipeline_version,
        "randomState": cfg.random_state,
        "preprocessMethods": list(cfg.preprocess_methods),
        "targetTransforms": list(cfg.target_transforms),
        "trainRatioList": list(cfg.train_ratio_list),
        "nRepeatedSplit": cfg.n_repeated_split,
        "algorithms": list(algorithms) if algorithms is not None else None,
        "featureSelection": {
            "enabled": cfg.enable_feature_selection,
            "candidateCounts": list(cfg.feature_count_candidates),
            "innerFolds": cfg.feature_selection_inner_folds,
            "innerRepeats": cfg.feature_selection_inner_repeats,
            "oneSE": cfg.feature_selection_one_se,
            "rankingPLSComponents": cfg.feature_ranking_pls_components,
            "minimumVariables": cfg.feature_selection_min_vars,
            "mode": cfg.feature_selection_mode,
            "fixedFeatureCount": cfg.fixed_feature_count,
        },
        "outlierScreening": {
            "enabled": cfg.enable_outlier_filter,
            "scope": cfg.outlier_filter_scope,
            "mccvNIter": cfg.mccv_n_iter,
            "maxOutlierRatio": cfg.max_outlier_ratio,
        },
        "complexityControl": {
            "maxPLSComponents": cfg.max_pls_components,
            "maxPCRComponents": cfg.max_pcr_components,
            "innerCVFolds": cfg.inner_cv_folds,
            "innerCVRepeats": cfg.inner_cv_repeats,
            "oneSE": cfg.use_one_standard_error_rule,
        },
        "searchLevel": cfg.search_level,
        "deepLearning": {
            "enabled": cfg.enable_deep_learning,
            "algorithms": list(cfg.deep_algorithms),
            "epochs": cfg.deep_epochs,
            "patience": cfg.deep_patience,
            "batchSize": cfg.deep_batch_size,
            "lr": cfg.deep_lr,
            "weightDecay": cfg.deep_weight_decay,
            "validationFraction": cfg.deep_validation_fraction,
        },
        "execution": {
            "fullSearch": bool(cfg.full_search),
            "strictLeakageGuard": bool(cfg.strict_leakage_guard),
            "sharedOuterSplits": bool(cfg.shared_outer_splits),
            "strictStackingNestedPreprocess": bool(cfg.strict_stacking_nested_preprocess),
            "outerSplitMethod": str(cfg.outer_split_method),
            "requireExact30Algorithms": bool(cfg.require_exact_30_algorithms),
            "requireCompleteFactorial": bool(cfg.require_complete_factorial),
            "useGPUForTorch": bool(cfg.use_gpu_for_torch),
            "parallelBackend": str(cfg.parallel_backend),
            "nJobs": int(cfg.n_jobs),
        },
        "predeclaredExclusionFile": cfg.predeclared_exclusion_file,
        "requireReasonForMissingReference": bool(cfg.require_reason_for_missing_reference),
        "provenanceLock": provenance,
    }


def run_signature(
    cfg: Config,
    algorithms: Optional[List[str]] = None,
    provenance: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Any]]:
    payload = scientific_config_payload(cfg, algorithms, provenance)
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest(), payload


def safe_rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return float("inf")
    return float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask])))


def safe_r2(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2 or np.std(y_true[mask]) < 1e-12:
        return float("nan")
    return float(r2_score(y_true[mask], y_pred[mask]))


def safe_mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return float("inf")
    return float(mean_absolute_error(y_true[mask], y_pred[mask]))


def set_global_seed(seed: int):
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def zscore_safe(X: np.ndarray):
    X = np.asarray(X, dtype=float)
    mu = np.nanmean(X, axis=0)
    sig = np.nanstd(X, axis=0, ddof=0)
    sig[sig < 1e-12] = 1.0
    Xs = (X - mu) / sig
    Xs[~np.isfinite(Xs)] = 0.0
    return Xs, mu, sig


def normalize_score(s: np.ndarray):
    s = np.asarray(s, dtype=float)
    s[~np.isfinite(s)] = 0.0
    mn, mx = np.min(s), np.max(s)
    if mx - mn < 1e-12:
        return np.zeros_like(s)
    return (s - mn) / (mx - mn)


def read_input_tables(spectral_file: str, phys_file: str):
    print(f"Reading spectral data: {spectral_file}")
    print(f"Reading reference data: {phys_file}")

    spectral_file = str(spectral_file)
    phys_file = str(phys_file)
    spec = pd.read_excel(spectral_file) if spectral_file.lower().endswith((".xlsx", ".xls")) else pd.read_csv(spectral_file)
    phys = pd.read_excel(phys_file) if phys_file.lower().endswith((".xlsx", ".xls")) else pd.read_csv(phys_file)

    wavelengths = pd.to_numeric(spec.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
    spec_sample_names = spec.columns[1:].astype(str).to_numpy()
    X_all = spec.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float).T

    target_names = phys.iloc[:, 0].astype(str).to_numpy()
    target_names = np.array([
        f"Analyte{i + 1}" if str(name).strip() == "" or str(name).lower() == "nan" else str(name).strip()
        for i, name in enumerate(target_names)
    ], dtype=str)
    phys_sample_names = phys.columns[1:].astype(str).to_numpy()
    Y_all = phys.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float).T

    spec_index = {name: i for i, name in enumerate(spec_sample_names)}
    phys_index = {name: i for i, name in enumerate(phys_sample_names)}
    common = [name for name in spec_sample_names if name in phys_index]
    if len(common) < 8:
        raise ValueError("Fewer than 8 samples can be aligned between the spectral and reference tables; modeling cannot proceed.")

    X = X_all[[spec_index[n] for n in common], :]
    Y = Y_all[[phys_index[n] for n in common], :]
    sample_names = np.asarray(common, dtype=str)

    print(f"Aligned sample count: {X.shape[0]}")
    print(f"Spectral variable count: {X.shape[1]}")
    print(f"Reference analyte count: {Y.shape[1]}\n")
    return wavelengths, sample_names, target_names, X, Y


def load_predeclared_exclusions(path: Optional[str]) -> pd.DataFrame:

    columns = ["target", "sampleName", "reason"]
    if path is None or str(path).strip() == "":
        return pd.DataFrame(columns=columns)
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Predeclared exclusion manifest does not exist: {p}")
    if p.suffix.lower() in {".xlsx", ".xls"}:
        raw = pd.read_excel(p)
    else:
        raw = pd.read_csv(p)
    normalized = {str(c).strip().lower(): c for c in raw.columns}

    def pick(candidates):
        for name in candidates:
            if name.lower() in normalized:
                return normalized[name.lower()]
        return None

    target_col = pick(["target", "analyte", "indicator"])
    sample_col = pick(["samplename", "sample", "sampleid"])
    reason_col = pick(["reason", "exclusionreason"])
    if target_col is None or sample_col is None:
        raise ValueError("The exclusion manifest must contain at least target and sampleName columns.")
    out_df = pd.DataFrame({
        "target": raw[target_col].astype(str).str.strip(),
        "sampleName": raw[sample_col].astype(str).str.strip(),
        "reason": "" if reason_col is None else raw[reason_col].fillna("").astype(str).str.strip(),
    })
    out_df = out_df[(out_df["target"] != "") & (out_df["sampleName"] != "")].copy()
    return out_df.reset_index(drop=True)


def resolve_target_population(
    target: str,
    y: np.ndarray,
    sample_names: np.ndarray,
    exclusions: pd.DataFrame,
    cfg: Config,
):

    y = np.asarray(y, dtype=float).ravel()
    sample_names = np.asarray(sample_names, dtype=str)
    finite = np.isfinite(y)
    declared = np.zeros(len(y), dtype=bool)
    declared_reason: Dict[str, str] = {}
    if exclusions is not None and not exclusions.empty:
        sub = exclusions[(exclusions["target"] == str(target)) | (exclusions["target"] == "*")]
        for _, row in sub.iterrows():
            name = str(row["sampleName"])
            declared[sample_names == name] = True
            declared_reason[name] = str(row.get("reason", "")).strip()

    missing_names = sample_names[~finite].astype(str).tolist()
    if cfg.require_reason_for_missing_reference:
        unresolved = [n for n in missing_names if not declared_reason.get(n, "")]
        if unresolved:
            raise RuntimeError(
                f"Analyte {target} has samples with missing reference values but no independent exclusion reason: {unresolved}"
            )

    excluded = (~finite) | declared
    keep = ~excluded
    audit_rows = []
    for i, name in enumerate(sample_names):
        if not excluded[i]:
            continue
        reasons = []
        if not finite[i]:
            reasons.append("MissingReferenceValue")
        if declared[i]:
            reasons.append(declared_reason.get(str(name), "PredeclaredIndependentQCExclusion") or
                           "PredeclaredIndependentQCExclusion")
        audit_rows.append({
            "target": str(target),
            "sampleName": str(name),
            "reason": "; ".join(reasons),
            "referenceValueFinite": bool(finite[i]),
        })

    population_meta = {
        "n_original_samples": int(len(y)),
        "n_presearch_outliers": int(excluded.sum()),
        "presearch_outlier_names": sample_names[excluded].astype(str).tolist(),
    }
    return keep, population_meta, audit_rows


def sg_smooth(X: np.ndarray, window: int = 17, polyorder: int = 2):
    n_var = X.shape[1]
    win = min(window, n_var)
    if win % 2 == 0:
        win -= 1
    if win < 5:
        return X.copy()
    return savgol_filter(X, window_length=win, polyorder=polyorder, axis=1, mode="interp")


def snv_preprocess(X: np.ndarray):
    mu = np.nanmean(X, axis=1, keepdims=True)
    sd = np.nanstd(X, axis=1, ddof=0, keepdims=True)
    sd[sd < 1e-12] = 1.0
    out = (X - mu) / sd
    out[~np.isfinite(out)] = 0.0
    return out


def msc_preprocess(X: np.ndarray):
    ref = np.nanmean(X, axis=0)
    out = np.zeros_like(X, dtype=float)
    for i in range(X.shape[0]):
        try:
            slope, intercept = np.polyfit(ref, X[i], 1)
            if abs(slope) < 1e-12:
                out[i] = X[i]
            else:
                out[i] = (X[i] - intercept) / slope
        except Exception:
            out[i] = X[i]
    out[~np.isfinite(out)] = 0.0
    return out


def detrend_preprocess(X: np.ndarray):
    x_axis = np.arange(X.shape[1], dtype=float)
    out = np.zeros_like(X, dtype=float)
    for i in range(X.shape[0]):
        try:
            p = np.polyfit(x_axis, X[i], 1)
            baseline = np.polyval(p, x_axis)
            out[i] = X[i] - baseline
        except Exception:
            out[i] = X[i]
    out[~np.isfinite(out)] = 0.0
    return out


def baseline_poly_preprocess(X: np.ndarray, degree: int = 2):
    x_axis = np.linspace(-1.0, 1.0, X.shape[1])
    out = np.zeros_like(X, dtype=float)
    for i in range(X.shape[0]):
        try:
            p = np.polyfit(x_axis, X[i], degree)
            baseline = np.polyval(p, x_axis)
            out[i] = X[i] - baseline
        except Exception:
            out[i] = X[i]
    out[~np.isfinite(out)] = 0.0
    return out


def first_derivative(X: np.ndarray, wavelengths: Optional[np.ndarray] = None):
    if wavelengths is None or len(wavelengths) != X.shape[1] or np.any(~np.isfinite(wavelengths)):
        out = np.gradient(X, axis=1)
    else:
        out = np.gradient(X, wavelengths, axis=1)
    out[~np.isfinite(out)] = 0.0
    return out


def second_derivative(X: np.ndarray, wavelengths: Optional[np.ndarray] = None):
    return first_derivative(first_derivative(X, wavelengths), wavelengths)


def vector_norm(X: np.ndarray):
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    norm[norm < 1e-12] = 1.0
    out = X / norm
    out[~np.isfinite(out)] = 0.0
    return out


def area_norm(X: np.ndarray):
    area = np.sum(np.abs(X), axis=1, keepdims=True)
    area[area < 1e-12] = 1.0
    out = X / area
    out[~np.isfinite(out)] = 0.0
    return out


def center_rows(X: np.ndarray):
    return X - np.nanmean(X, axis=1, keepdims=True)


def _safe_savgol_derivative(X: np.ndarray, deriv: int, wavelengths: Optional[np.ndarray] = None):
    n_var = X.shape[1]
    win = min(17, n_var if n_var % 2 == 1 else n_var - 1)
    if win < 5:
        return first_derivative(X, wavelengths) if deriv == 1 else second_derivative(X, wavelengths)
    delta = 1.0
    if wavelengths is not None:
        w = np.asarray(wavelengths, dtype=float).ravel()
        if w.size == n_var and np.all(np.isfinite(w)):
            dw = np.diff(w)
            dw = dw[np.isfinite(dw) & (np.abs(dw) > 1e-12)]
            if dw.size:
                delta = float(np.median(np.abs(dw)))
    return savgol_filter(
        X, window_length=win, polyorder=2, deriv=deriv,
        delta=delta, axis=1, mode="interp"
    )


def _msc_with_reference(X: np.ndarray, ref: np.ndarray):
    X = np.asarray(X, dtype=float)
    ref = np.asarray(ref, dtype=float).ravel()
    out = np.zeros_like(X, dtype=float)
    for i in range(X.shape[0]):
        try:
            slope, intercept = np.polyfit(ref, X[i], 1)
            out[i] = X[i] if abs(slope) < 1e-12 else (X[i] - intercept) / slope
        except Exception:
            out[i] = X[i]
    out[~np.isfinite(out)] = 0.0
    return out


_PREPROCESS_STEPS = {
    "RAW": (),
    "CENTER": ("CENTER",),
    "STANDARDIZE": ("STANDARDIZE",),
    "MINMAX": ("MINMAX",),
    "ROBUST": ("ROBUST",),
    "VECTOR_NORM": ("VECTOR_NORM",),
    "AREA_NORM": ("AREA_NORM",),
    "SG": ("SG",),
    "SG1": ("SG1",),
    "SG2": ("SG2",),
    "MSC": ("MSC",),
    "SNV": ("SNV",),
    "DETREND": ("DETREND",),
    "BASELINE_POLY2": ("BASELINE_POLY2",),
    "BASELINE_POLY3": ("BASELINE_POLY3",),
    "SNV-SG": ("SNV", "SG"),
    "MSC-SG": ("MSC", "SG"),
    "SG-SNV": ("SG", "SNV"),
    "SG-MSC": ("SG", "MSC"),
    "DE1": ("DE1",),
    "DE2": ("DE2",),
    "DE1-SNV": ("DE1", "SNV"),
    "DE1-MSC": ("DE1", "MSC"),
    "DE2-SNV": ("DE2", "SNV"),
    "DE2-MSC": ("DE2", "MSC"),
    "BASELINE_POLY2-SNV": ("BASELINE_POLY2", "SNV"),
    "BASELINE_POLY2-MSC": ("BASELINE_POLY2", "MSC"),
    "BASELINE_POLY2-SG": ("BASELINE_POLY2", "SG"),
}


class SpectralPreprocessor:


    def __init__(self, method: str, wavelengths: np.ndarray):
        self.method = str(method).upper()
        if self.method not in _PREPROCESS_STEPS:
            raise ValueError(f"Unknown preprocessing method: {method}")
        self.steps = _PREPROCESS_STEPS[self.method]
        self.wavelengths = np.asarray(wavelengths, dtype=float)
        self.states_: List[Optional[Dict[str, np.ndarray]]] = []
        self.fitted_ = False

    def _step(self, X: np.ndarray, step: str, state=None, fit: bool = False):
        X = np.asarray(X, dtype=float)
        if step == "CENTER":
            if fit:
                state = {"mu": np.nanmean(X, axis=0)}
            return X - state["mu"], state
        if step == "STANDARDIZE":
            if fit:
                mu = np.nanmean(X, axis=0)
                scale = np.nanstd(X, axis=0, ddof=0)
                scale[scale < 1e-12] = 1.0
                state = {"mu": mu, "scale": scale}
            out = (X - state["mu"]) / state["scale"]
            return out, state
        if step == "MINMAX":
            if fit:
                mn = np.nanmin(X, axis=0)
                mx = np.nanmax(X, axis=0)
                scale = mx - mn
                scale[scale < 1e-12] = 1.0
                state = {"min": mn, "scale": scale}
            return (X - state["min"]) / state["scale"], state
        if step == "ROBUST":
            if fit:
                med = np.nanmedian(X, axis=0)
                q25 = np.nanpercentile(X, 25, axis=0)
                q75 = np.nanpercentile(X, 75, axis=0)
                scale = q75 - q25
                scale[scale < 1e-12] = 1.0
                state = {"center": med, "scale": scale}
            return (X - state["center"]) / state["scale"], state
        if step == "VECTOR_NORM":
            return vector_norm(X), None
        if step == "AREA_NORM":
            return area_norm(X), None
        if step == "SG":
            return sg_smooth(X, window=17, polyorder=2), None
        if step == "SG1":
            return _safe_savgol_derivative(X, deriv=1, wavelengths=self.wavelengths), None
        if step == "SG2":
            return _safe_savgol_derivative(X, deriv=2, wavelengths=self.wavelengths), None
        if step == "MSC":
            if fit:
                state = {"ref": np.nanmean(X, axis=0)}
            return _msc_with_reference(X, state["ref"]), state
        if step == "SNV":
            return snv_preprocess(X), None
        if step == "DETREND":
            return detrend_preprocess(X), None
        if step == "BASELINE_POLY2":
            return baseline_poly_preprocess(X, degree=2), None
        if step == "BASELINE_POLY3":
            return baseline_poly_preprocess(X, degree=3), None
        if step == "DE1":
            return first_derivative(X, self.wavelengths), None
        if step == "DE2":
            return second_derivative(X, self.wavelengths), None
        raise ValueError(f"Unknown preprocessing step: {step}")

    def fit_transform(self, X: np.ndarray):
        out = np.asarray(X, dtype=float).copy()
        self.states_ = []
        for step in self.steps:
            out, state = self._step(out, step, fit=True)
            out[~np.isfinite(out)] = 0.0
            self.states_.append(state)
        self.fitted_ = True
        return out

    def transform(self, X: np.ndarray):
        if not self.fitted_:
            raise RuntimeError("SpectralPreprocessor has not been fitted on the calibration set.")
        out = np.asarray(X, dtype=float).copy()
        for step, state in zip(self.steps, self.states_):
            out, _ = self._step(out, step, state=state, fit=False)
            out[~np.isfinite(out)] = 0.0
        return out


def apply_preprocess(X: np.ndarray, wavelengths: np.ndarray, method: str):

    return SpectralPreprocessor(method, wavelengths).fit_transform(X)


def remove_bad_variables_train_test(X_train: np.ndarray, X_test: np.ndarray):

    good = np.all(np.isfinite(X_train), axis=0) & (np.nanstd(X_train, axis=0) > 1e-12)
    return X_train[:, good], X_test[:, good], good


def remove_bad_variables(X: np.ndarray):
    good = np.all(np.isfinite(X), axis=0) & (np.nanstd(X, axis=0) > 1e-12)
    return X[:, good], good


def can_apply_y_transform(y: np.ndarray, method: str):
    method = method.lower()
    y = np.asarray(y, dtype=float)
    if method in {"raw", "standard"}:
        return bool(np.all(np.isfinite(y)) and np.ptp(y) > 0)
    if method in {"log1p", "sqrt"}:
        return bool(np.all(np.isfinite(y)) and np.all(y >= 0) and np.ptp(y) > 0)
    return False


class TargetTransformer:


    def __init__(self, method: str):
        self.method = str(method).lower()
        self.fitted_ = False

    def fit(self, y_train: np.ndarray):
        y = np.asarray(y_train, dtype=float).ravel()
        if not can_apply_y_transform(y, self.method):
            raise ValueError(f"Calibration y does not satisfy the requirements of transformation {self.method}")
        if self.method == "standard":
            self.mu_ = float(np.mean(y))
            self.sd_ = float(np.std(y)) if np.std(y) > 1e-12 else 1.0
        self.fitted_ = True
        return self

    def transform(self, y: np.ndarray):
        if not self.fitted_:
            raise RuntimeError("TargetTransformer has not been fitted on calibration y.")
        y = np.asarray(y, dtype=float)
        if self.method == "raw":
            return y.copy()
        if self.method == "log1p":
            return np.log1p(y)
        if self.method == "sqrt":
            return np.sqrt(y)
        if self.method == "standard":
            return (y - self.mu_) / self.sd_
        raise ValueError(f"Unknown response transformation: {self.method}")

    def inverse_transform(self, z: np.ndarray):

        if not self.fitted_:
            raise RuntimeError("TargetTransformer has not been fitted.")
        z = np.asarray(z, dtype=float)
        if self.method == "raw":
            return z.copy()
        if self.method == "log1p":
            return np.maximum(np.expm1(z), 0.0)
        if self.method == "sqrt":
            return np.square(np.maximum(z, 0.0))
        if self.method == "standard":
            return z * self.sd_ + self.mu_
        raise ValueError(f"Unknown response transformation: {self.method}")


def transform_y(y: np.ndarray, method: str):

    tr = TargetTransformer(method).fit(y)
    return tr.transform(y), tr.inverse_transform


def mccv_outlier_keep(X: np.ndarray, y: np.ndarray, cfg: Config, random_state: Optional[int] = None):

    n = X.shape[0]
    keep = np.ones(n, dtype=bool)
    if not cfg.enable_outlier_filter or n < 15:
        return keep

    seed = cfg.random_state if random_state is None else int(random_state)
    rng = np.random.default_rng(seed)
    pred_err = []
    successful = 0
    attempts = 0
    max_attempts = max(cfg.mccv_n_iter * 5, cfg.mccv_n_iter + 20)

    while successful < cfg.mccv_n_iter and attempts < max_attempts:
        attempts += 1
        idx = rng.permutation(n)
        n_train = max(8, int(round(0.78 * n)))
        tr, te = idx[:n_train], idx[n_train:]
        if len(te) < 3:
            continue
        try:
            n_comp = min(12, X.shape[1], len(tr) - 2)
            if n_comp < 1:
                continue
            model = PLSRegression(n_components=n_comp, scale=True)
            model.fit(X[tr], y[tr])
            row = np.full(n, np.nan, dtype=float)
            row[te] = np.abs(y[te] - model.predict(X[te]).ravel())
            pred_err.append(row)
            successful += 1
        except Exception:
            continue

    if successful != cfg.mccv_n_iter:
        raise RuntimeError(
            f"MCCV-like screening completed only {successful}/{cfg.mccv_n_iter} successful submodels; "
            "incomplete screening results are rejected to preserve consistency with the manuscript method."
        )

    pred_err = np.asarray(pred_err, dtype=float)
    counts = np.sum(np.isfinite(pred_err), axis=0)
    if np.any(counts == 0):
        raise RuntimeError("MCCV-like screening contains samples that never entered a temporary prediction set.")

    mean_err = np.nanmean(pred_err, axis=0)
    med = np.nanmedian(mean_err)
    mad = np.nanmedian(np.abs(mean_err - med))
    threshold = med + 3.0 * mad
    outlier = mean_err > threshold
    max_outliers = max(10, int(round(cfg.max_outlier_ratio * n)))
    if np.sum(~outlier) >= 8 and np.sum(outlier) <= max_outliers:
        keep = ~outlier
    return keep


def _repeated_kfold_splits(n_samples: int, n_splits: int, n_repeats: int, random_state: int):

    n_splits = max(2, min(int(n_splits), int(n_samples)))
    for rep in range(max(1, int(n_repeats))):
        seed = stable_seed(random_state, "inner_cv", rep)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold_id, (tr, va) in enumerate(kf.split(np.arange(n_samples)), start=1):
            yield rep, fold_id, np.asarray(tr, dtype=int), np.asarray(va, dtype=int)


def _feature_scores_pls_corr(X: np.ndarray, y: np.ndarray, cfg: Config) -> np.ndarray:

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    p = X.shape[1]
    n_comp = max(1, min(cfg.feature_ranking_pls_components, X.shape[0] - 2, p))
    coef_score = np.zeros(p, dtype=float)
    try:
        pls = PLSRegression(n_components=n_comp, scale=True).fit(X, y)
        coef = np.abs(np.asarray(pls.coef_).ravel())
        if coef.size == p and np.all(np.isfinite(coef)):
            coef_score = coef
    except Exception:
        pass

    corr_score = np.zeros(p, dtype=float)
    yc = y - np.mean(y)
    yss = float(np.sqrt(np.sum(yc * yc)))
    if yss > 1e-12:
        Xc = X - np.mean(X, axis=0, keepdims=True)
        xss = np.sqrt(np.sum(Xc * Xc, axis=0))
        valid = xss > 1e-12
        corr_score[valid] = np.abs((Xc[:, valid].T @ yc) / (xss[valid] * yss))
        corr_score[~np.isfinite(corr_score)] = 0.0
    return 0.70 * normalize_score(coef_score) + 0.30 * normalize_score(corr_score)


def _feature_rank_pls_corr(X: np.ndarray, y: np.ndarray, cfg: Config) -> np.ndarray:
    return np.argsort(_feature_scores_pls_corr(X, y, cfg))[::-1]


def _select_contiguous_feature_window(X: np.ndarray, y: np.ndarray, count: int, cfg: Config) -> np.ndarray:

    p = int(X.shape[1])
    k = max(1, min(int(count), p))
    if k >= p:
        return np.arange(p, dtype=int)
    scores = _feature_scores_pls_corr(X, y, cfg)
    window_scores = np.convolve(scores, np.ones(k, dtype=float), mode="valid")
    start = int(np.nanargmax(window_scores))
    return np.arange(start, start + k, dtype=int)

def _valid_feature_counts(p: int, n: int, cfg: Config) -> List[int]:
    if str(cfg.feature_selection_mode).lower() == "fixed_count_nested":
        return [max(1, min(int(cfg.fixed_feature_count), int(p)))]
    counts: List[int] = []
    for value in cfg.feature_count_candidates:
        k = p if int(value) < 0 else int(value)
        k = max(int(cfg.feature_selection_min_vars), min(k, p))
        if k not in counts:
            counts.append(k)
    if p not in counts:
        counts.append(p)
    return sorted(set(counts))


def _select_one_se(means: np.ndarray, ses: np.ndarray, complexities: List[float], enabled: bool) -> int:

    means = np.asarray(means, dtype=float)
    ses = np.asarray(ses, dtype=float)
    finite = np.isfinite(means)
    if not np.any(finite):
        raise RuntimeError("No finite inner-CV candidate score.")
    best = int(np.nanargmin(means))
    if not enabled:
        return best
    threshold = means[best] + (ses[best] if np.isfinite(ses[best]) else 0.0)
    acceptable = [i for i, m in enumerate(means) if np.isfinite(m) and m <= threshold]
    return min(acceptable, key=lambda i: (complexities[i], means[i]))


def adaptive_select_features(
    X: np.ndarray,
    y_raw: np.ndarray,
    target_method: str,
    cfg: Config,
    random_state: Optional[int] = None,
) -> np.ndarray:

    X = np.asarray(X, dtype=float)
    y_raw = np.asarray(y_raw, dtype=float).ravel()
    p = X.shape[1]
    if not cfg.enable_feature_selection or p <= cfg.feature_selection_min_vars:
        return np.arange(p, dtype=int)
    if str(cfg.feature_selection_mode).lower() == "fixed_count_nested":
        rank = _feature_rank_pls_corr(X, TargetTransformer(target_method).fit(y_raw).transform(y_raw), cfg)
        return np.sort(rank[:min(int(cfg.fixed_feature_count), p)]).astype(int)

    seed = cfg.random_state if random_state is None else int(random_state)
    candidates = _valid_feature_counts(p, len(y_raw), cfg)
    fold_errors: Dict[int, List[float]] = {k: [] for k in candidates}

    for rep, fold_id, tr, va in _repeated_kfold_splits(
        len(y_raw), cfg.feature_selection_inner_folds,
        cfg.feature_selection_inner_repeats, seed,
    ):
        transformer = TargetTransformer(target_method).fit(y_raw[tr])
        ytr = transformer.transform(y_raw[tr])
        rank = _feature_rank_pls_corr(X[tr], ytr, cfg)
        for k in candidates:
            idx = np.sort(rank[:k]) if k < p else np.arange(p, dtype=int)
            n_comp = max(1, min(cfg.feature_ranking_pls_components, len(tr) - 2, len(idx)))
            try:
                mdl = PLSRegression(n_components=n_comp, scale=True).fit(X[tr][:, idx], ytr)
                pred_work = mdl.predict(X[va][:, idx]).ravel()
                pred_raw = transformer.inverse_transform(pred_work)
                fold_errors[k].append(safe_rmse(y_raw[va], pred_raw))
            except Exception:
                fold_errors[k].append(float("inf"))

    means, ses = [], []
    for k in candidates:
        vals = np.asarray(fold_errors[k], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            means.append(float("inf")); ses.append(float("inf"))
        else:
            means.append(float(np.mean(vals)))
            ses.append(float(np.std(vals, ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0)

    chosen_pos = _select_one_se(
        np.asarray(means), np.asarray(ses), [float(k) for k in candidates],
        cfg.feature_selection_one_se,
    )
    chosen_k = int(candidates[chosen_pos])

    full_transformer = TargetTransformer(target_method).fit(y_raw)
    y_full = full_transformer.transform(y_raw)
    final_rank = _feature_rank_pls_corr(X, y_full, cfg)
    if chosen_k >= p:
        return np.arange(p, dtype=int)
    return np.sort(final_rank[:chosen_k]).astype(int)


def adaptive_feature_count_from_raw(
    X_raw: np.ndarray,
    y_raw: np.ndarray,
    target_method: str,
    preprocess_method: str,
    wavelengths: np.ndarray,
    cfg: Config,
    random_state: Optional[int] = None,
) -> int:

    X_raw = np.asarray(X_raw, dtype=float)
    y_raw = np.asarray(y_raw, dtype=float).ravel()
    if not cfg.enable_feature_selection:
        return int(X_raw.shape[1])
    if str(cfg.feature_selection_mode).lower() == "fixed_count_nested":
        return max(1, min(int(cfg.fixed_feature_count), int(X_raw.shape[1])))
    seed = cfg.random_state if random_state is None else int(random_state)
    candidates = _valid_feature_counts(X_raw.shape[1], len(y_raw), cfg)
    fold_errors: Dict[int, List[float]] = {k: [] for k in candidates}

    for _, _, tr, va in _repeated_kfold_splits(
        len(y_raw), cfg.feature_selection_inner_folds,
        cfg.feature_selection_inner_repeats, seed,
    ):
        try:
            pre = SpectralPreprocessor(preprocess_method, wavelengths)
            Xtr = pre.fit_transform(X_raw[tr])
            Xva = pre.transform(X_raw[va])
            Xtr, Xva, _ = remove_bad_variables_train_test(Xtr, Xva)
            transformer = TargetTransformer(target_method).fit(y_raw[tr])
            ytr = transformer.transform(y_raw[tr])
            rank = _feature_rank_pls_corr(Xtr, ytr, cfg)
            p_fold = Xtr.shape[1]
            for k in candidates:
                kk = min(int(k), p_fold)
                idx = np.sort(rank[:kk]) if kk < p_fold else np.arange(p_fold, dtype=int)
                n_comp = max(1, min(cfg.feature_ranking_pls_components, len(tr) - 2, len(idx)))
                mdl = PLSRegression(n_components=n_comp, scale=True).fit(Xtr[:, idx], ytr)
                pred_raw = transformer.inverse_transform(mdl.predict(Xva[:, idx]).ravel())
                fold_errors[k].append(safe_rmse(y_raw[va], pred_raw))
        except Exception:
            for k in candidates:
                fold_errors[k].append(float("inf"))

    means, ses = [], []
    for k in candidates:
        vals = np.asarray(fold_errors[k], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            means.append(float("inf")); ses.append(float("inf"))
        else:
            means.append(float(np.mean(vals)))
            ses.append(float(np.std(vals, ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0)
    pos = _select_one_se(
        np.asarray(means), np.asarray(ses), [float(k) for k in candidates],
        cfg.feature_selection_one_se,
    )
    return int(candidates[pos])


def cars_select_features(X: np.ndarray, y: np.ndarray, cfg: Config, random_state: Optional[int] = None):
    return adaptive_select_features(X, np.asarray(y, dtype=float), "raw", cfg, random_state)


def kennard_stone_split(X: np.ndarray, train_ratio: float):
    n = X.shape[0]
    n_train = max(2, int(round(train_ratio * n)))
    n_train = min(n_train, n - 1)
    Xs, _, _ = zscore_safe(X)
    D = squareform(pdist(Xs, metric="euclidean"))
    i1, i2 = np.unravel_index(np.argmax(D), D.shape)
    selected = [int(i1), int(i2)]
    while len(selected) < n_train:
        remain = np.setdiff1d(np.arange(n), np.asarray(selected), assume_unique=False)
        min_dist = np.min(D[remain][:, selected], axis=1)
        selected.append(int(remain[np.argmax(min_dist)]))
    train_idx = np.asarray(selected, dtype=int)
    test_idx = np.setdiff1d(np.arange(n), train_idx, assume_unique=False)
    return train_idx, test_idx


def hybrid_split(X: np.ndarray, y: np.ndarray, train_ratio: float, rep: int, random_state: int):
    n = X.shape[0]
    n_train = max(2, int(round(train_ratio * n)))
    n_train = min(n_train, n - 1)
    rng = np.random.default_rng(random_state + rep)
    order = np.argsort(y)
    n_bins = min(5, max(2, n // 12))
    bins = []
    for b in range(n_bins):
        s = int(math.floor(b * n / n_bins))
        e = int(math.floor((b + 1) * n / n_bins))
        bins.append(order[s:e])
    train_idx: List[int] = []
    for ids in bins:
        ids = ids.copy()
        rng.shuffle(ids)
        nb = max(1, int(round(train_ratio * len(ids))))
        train_idx.extend(ids[:nb].tolist())
    train_idx = np.unique(np.asarray(train_idx, dtype=int))
    if len(train_idx) > n_train:
        train_idx = rng.choice(train_idx, size=n_train, replace=False)
    elif len(train_idx) < n_train:
        rest = np.setdiff1d(np.arange(n), train_idx)
        add = rng.choice(rest, size=n_train - len(train_idx), replace=False)
        train_idx = np.unique(np.concatenate([train_idx, add]))
    test_idx = np.setdiff1d(np.arange(n), train_idx)
    if len(test_idx) < 3:
        return kennard_stone_split(X, train_ratio)
    return train_idx, test_idx


def build_shared_outer_splits(X: np.ndarray, y: np.ndarray, cfg: Config) -> List[SplitSpec]:

    if str(cfg.outer_split_method).lower() != "response_stratified_random":
        raise ValueError("The manuscript workflow only permits outer_split_method='response_stratified_random'.")
    specs: List[SplitSpec] = []
    split_id = 0
    for ratio in cfg.train_ratio_list:
        for rep in range(1, cfg.n_repeated_split + 1):
            split_id += 1
            split_seed = stable_seed(cfg.random_state, "outer_split", f"{ratio:.6f}", rep)
            train_idx, test_idx = hybrid_split(X, y, ratio, rep, split_seed)
            train_idx = np.asarray(train_idx, dtype=int)
            test_idx = np.asarray(test_idx, dtype=int)
            if len(test_idx) < 5 or np.std(y[test_idx]) < 1e-12:
                raise RuntimeError(
                    f"Invalid outer split: ratio={ratio}, rep={rep}, prediction set is too small or y has no variation"
                )
            if len(train_idx) < 8:
                raise RuntimeError(
                    f"Invalid outer split: ratio={ratio}, rep={rep}, calibration set has too few samples"
                )
            specs.append(SplitSpec(
                split_id=split_id,
                train_ratio=float(ratio),
                repeat_id=int(rep),
                train_idx_raw=train_idx.copy(),
                train_idx=train_idx.copy(),
                test_idx=test_idx,
                n_calibration_outliers=0,
            ))
    return specs


def legacy_configuration_split_id(
    cfg: Config,
    y_transform: str,
    preprocess: str,
    train_ratio: float,
    repeat_id: int,
) -> int:

    yi = list(cfg.target_transforms).index(y_transform)
    pi = list(cfg.preprocess_methods).index(preprocess)
    ri = list(cfg.train_ratio_list).index(float(train_ratio))
    return int((((yi * len(cfg.preprocess_methods) + pi) * len(cfg.train_ratio_list) + ri)
                * cfg.n_repeated_split) + int(repeat_id))


class RBFLSSVMRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, gammas=None, sigmas=None, cv=5, random_state=2026):
        self.gammas = gammas
        self.sigmas = sigmas
        self.cv = cv
        self.random_state = random_state

    @staticmethod
    def _rbf_kernel(X1, X2, sigma2):
        D2 = cdist(X1, X2, metric="sqeuclidean")
        return np.exp(-D2 / max(2.0 * sigma2, 1e-12))

    def _fit_core(self, X, y, gamma, sigma2):
        K = self._rbf_kernel(X, X, sigma2)
        n = X.shape[0]
        A = np.block([[np.zeros((1, 1)), np.ones((1, n))],
                      [np.ones((n, 1)), K + np.eye(n) / gamma]])
        b = np.concatenate([[0.0], y.ravel()])
        sol = np.linalg.solve(A + 1e-10 * np.eye(n + 1), b)
        return sol[0], sol[1:]

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        gammas = self.gammas if self.gammas is not None else np.logspace(-2, 4, 5)
        sigmas = self.sigmas if self.sigmas is not None else np.logspace(-2, 4, 5)
        kf = KFold(n_splits=min(self.cv, len(y)), shuffle=True, random_state=self.random_state)
        best_rmse = float("inf")
        best_params = None

        for gamma in gammas:
            for sigma2 in sigmas:
                pred = np.full(len(y), np.nan, dtype=float)
                for tr, te in kf.split(X):
                    try:
                        scaler = StandardScaler().fit(X[tr])
                        Xtr = scaler.transform(X[tr])
                        Xte = scaler.transform(X[te])
                        y_mu = float(np.mean(y[tr]))
                        y_sig = float(np.std(y[tr])) if np.std(y[tr]) > 1e-12 else 1.0
                        ytr = (y[tr] - y_mu) / y_sig
                        b0, alpha = self._fit_core(Xtr, ytr, gamma, sigma2)
                        z = self._rbf_kernel(Xte, Xtr, sigma2) @ alpha + b0
                        pred[te] = z * y_sig + y_mu
                    except Exception:
                        pred[te] = np.nan
                rmse = complete_oof_rmse(y, pred)
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_params = (float(gamma), float(sigma2))

        if best_params is None:
            raise RuntimeError("RBF-LS-SVM has no parameter combination that completed all inner-CV folds.")

        self.gamma_, self.sigma2_ = best_params
        self.scaler_ = StandardScaler().fit(X)
        self.y_mu_ = float(np.mean(y))
        self.y_sig_ = float(np.std(y)) if np.std(y) > 1e-12 else 1.0
        Xs = self.scaler_.transform(X)
        ys = (y - self.y_mu_) / self.y_sig_
        self.b_, self.alpha_ = self._fit_core(Xs, ys, self.gamma_, self.sigma2_)
        self.X_train_ = Xs
        return self

    def predict(self, X):
        Xs = self.scaler_.transform(np.asarray(X, dtype=float))
        ys = self._rbf_kernel(Xs, self.X_train_, self.sigma2_) @ self.alpha_ + self.b_
        return ys * self.y_sig_ + self.y_mu_


class ELMRegressor(BaseEstimator, RegressorMixin):


    def __init__(self, hidden_list=(20, 40, 80, 120),
                 lambdas=(1e-4, 1e-3, 1e-2, 1e-1),
                 cv=5, random_state=2026):
        self.hidden_list = hidden_list
        self.lambdas = lambdas
        self.cv = cv
        self.random_state = random_state

    @staticmethod
    def _solve_beta(H, y, lam):
        A = H.T @ H + lam * np.eye(H.shape[1])
        b = H.T @ y
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(A) @ b

    def _weights(self, n_features, h):
        rng = np.random.default_rng(stable_seed(self.random_state, "ELM", h, n_features))
        return rng.normal(size=(n_features, h)), rng.normal(size=(h,))

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        kf = KFold(n_splits=min(self.cv, len(y)), shuffle=True, random_state=self.random_state)
        best_rmse = float("inf")
        best_params = None

        for h in self.hidden_list:
            W, b = self._weights(X.shape[1], int(h))
            for lam in self.lambdas:
                pred = np.full(len(y), np.nan, dtype=float)
                for tr, te in kf.split(X):
                    try:
                        scaler = StandardScaler().fit(X[tr])
                        Xtr = scaler.transform(X[tr])
                        Xte = scaler.transform(X[te])
                        y_mu = float(np.mean(y[tr]))
                        y_sig = float(np.std(y[tr])) if np.std(y[tr]) > 1e-12 else 1.0
                        ytr = (y[tr] - y_mu) / y_sig
                        Htr = np.tanh(Xtr @ W + b)
                        Hte = np.tanh(Xte @ W + b)
                        beta = self._solve_beta(Htr, ytr, float(lam))
                        pred[te] = (Hte @ beta) * y_sig + y_mu
                    except Exception:
                        pred[te] = np.nan
                rmse = complete_oof_rmse(y, pred)
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_params = (int(h), float(lam))

        if best_params is None:
            raise RuntimeError("ELM has no parameter combination that completed all inner-CV folds.")

        self.hidden_, self.lambda_ = best_params
        self.scaler_ = StandardScaler().fit(X)
        self.y_mu_ = float(np.mean(y))
        self.y_sig_ = float(np.std(y)) if np.std(y) > 1e-12 else 1.0
        Xs = self.scaler_.transform(X)
        ys = (y - self.y_mu_) / self.y_sig_
        self.W_, self.b_ = self._weights(X.shape[1], self.hidden_)
        H = np.tanh(Xs @ self.W_ + self.b_)
        self.beta_ = self._solve_beta(H, ys, self.lambda_)
        return self

    def predict(self, X):
        Xs = self.scaler_.transform(np.asarray(X, dtype=float))
        H = np.tanh(Xs @ self.W_ + self.b_)
        return (H @ self.beta_) * self.y_sig_ + self.y_mu_


class PCRRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, max_components=25, cv=5, random_state=2026):
        self.max_components = max_components
        self.cv = cv
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        max_comp = max(1, min(self.max_components, X.shape[1], X.shape[0] - 2))
        kf = KFold(n_splits=min(self.cv, len(y)), shuffle=True, random_state=self.random_state)
        best_comp, best_rmse = None, float("inf")

        for a in range(1, max_comp + 1):
            pred = np.full(len(y), np.nan, dtype=float)
            for tr, te in kf.split(X):
                try:
                    scaler = StandardScaler().fit(X[tr])
                    Xtr = scaler.transform(X[tr])
                    Xte = scaler.transform(X[te])
                    pca = PCA(n_components=min(a, len(tr) - 1, X.shape[1]), random_state=self.random_state)
                    Ttr = pca.fit_transform(Xtr)
                    Tte = pca.transform(Xte)
                    reg = Ridge(alpha=1e-8).fit(Ttr, y[tr])
                    pred[te] = reg.predict(Tte)
                except Exception:
                    pred[te] = np.nan
            rmse = complete_oof_rmse(y, pred)
            if rmse < best_rmse:
                best_rmse, best_comp = rmse, a

        if best_comp is None:
            raise RuntimeError("PCR has no component count that completed all inner-CV folds.")

        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        self.pca_ = PCA(n_components=best_comp, random_state=self.random_state).fit(Xs)
        self.reg_ = Ridge(alpha=1e-8).fit(self.pca_.transform(Xs), y)
        self.n_components_ = int(best_comp)
        return self

    def predict(self, X):
        Xs = self.scaler_.transform(np.asarray(X, dtype=float))
        return self.reg_.predict(self.pca_.transform(Xs))


class PLSLSSVMRegressor(BaseEstimator, RegressorMixin):


    def __init__(self, max_components=20, cv=3, random_state=2026, gammas=None, sigmas=None):
        self.max_components = max_components
        self.cv = cv
        self.random_state = random_state
        self.gammas = gammas
        self.sigmas = sigmas

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()


        local_cfg = Config(
            max_pls_components=self.max_components,
            inner_cv_folds=max(3, self.cv),
            inner_cv_repeats=1,
            use_one_standard_error_rule=True,
        )
        pls = fit_best_pls(
            X, y, local_cfg, random_state=self.random_state,
            y_raw=y, target_method="raw",
        )
        best_comp = int(pls.best_n_components_)

        gammas = self.gammas if self.gammas is not None else np.logspace(-2, 4, 5)
        sigmas = self.sigmas if self.sigmas is not None else np.logspace(-2, 4, 5)
        kf = KFold(n_splits=min(self.cv, len(y)), shuffle=True, random_state=self.random_state)
        folds = list(kf.split(X))


        fold_latent = []
        for fold_id, (tr, te) in enumerate(folds, start=1):
            n_comp_fold = max(1, min(best_comp, len(tr) - 2, X.shape[1]))
            pls_fold = PLSRegression(n_components=n_comp_fold, scale=True).fit(X[tr], y[tr])
            fold_latent.append((tr, te, pls_fold.transform(X[tr]), pls_fold.transform(X[te])))

        best_rmse = float("inf")
        best_params = None
        helper = RBFLSSVMRegressor(cv=self.cv, random_state=self.random_state)

        for gamma in gammas:
            for sigma2 in sigmas:
                pred = np.full(len(y), np.nan, dtype=float)
                for tr, te, Ttr_raw, Tte_raw in fold_latent:
                    try:
                        scaler = StandardScaler().fit(Ttr_raw)
                        Ttr = scaler.transform(Ttr_raw)
                        Tte = scaler.transform(Tte_raw)
                        y_mu = float(np.mean(y[tr]))
                        y_sig = float(np.std(y[tr])) if np.std(y[tr]) > 1e-12 else 1.0
                        ytr = (y[tr] - y_mu) / y_sig
                        b0, alpha = helper._fit_core(Ttr, ytr, gamma, sigma2)
                        z = helper._rbf_kernel(Tte, Ttr, sigma2) @ alpha + b0
                        pred[te] = z * y_sig + y_mu
                    except Exception:
                        pred[te] = np.nan
                rmse = complete_oof_rmse(y, pred)
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_params = (float(gamma), float(sigma2))

        if best_params is None:
            raise RuntimeError("PLS-LSSVM has no parameter combination that completed all inner-CV folds.")

        self.n_components_ = best_comp
        self.gamma_, self.sigma2_ = best_params
        self.pls_ = PLSRegression(n_components=best_comp, scale=True).fit(X, y)
        T = self.pls_.transform(X)
        self.lssvm_ = RBFLSSVMRegressor(
            gammas=[self.gamma_], sigmas=[self.sigma2_], cv=2,
            random_state=self.random_state,
        ).fit(T, y)
        return self

    def predict(self, X):
        return self.lssvm_.predict(self.pls_.transform(np.asarray(X, dtype=float)))


if TORCH_AVAILABLE:
    class TorchMLPNet(nn.Module):
        def __init__(self, in_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.15),
                nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.10),
                nn.Linear(64, 1)
            )
        def forward(self, x):
            return self.net(x).squeeze(-1)

    class TorchCNN1DNet(nn.Module):
        def __init__(self, in_dim):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=9, padding=4), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(16, 32, kernel_size=7, padding=3), nn.ReLU(), nn.AdaptiveAvgPool1d(16)
            )
            self.fc = nn.Sequential(nn.Flatten(), nn.Linear(32 * 16, 64), nn.ReLU(), nn.Dropout(0.10), nn.Linear(64, 1))
        def forward(self, x):
            x = x.unsqueeze(1)
            return self.fc(self.conv(x)).squeeze(-1)

    class TorchCNNGRUNet(nn.Module):
        def __init__(self, in_dim):
            super().__init__()
            self.conv = nn.Sequential(nn.Conv1d(1, 16, kernel_size=9, padding=4), nn.ReLU(), nn.MaxPool1d(4))
            self.gru = nn.GRU(input_size=16, hidden_size=24, batch_first=True)
            self.fc = nn.Sequential(nn.Linear(24, 32), nn.ReLU(), nn.Dropout(0.10), nn.Linear(32, 1))
        def forward(self, x):
            z = self.conv(x.unsqueeze(1)).transpose(1, 2)
            out, _ = self.gru(z)
            return self.fc(out[:, -1, :]).squeeze(-1)

class TorchRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, arch="mlp", epochs=180, patience=25, batch_size=16,
                 lr=1e-3, weight_decay=1e-3, random_state=2026,
                 use_gpu=True, validation_fraction=0.20):
        self.arch = arch
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.use_gpu = use_gpu
        self.validation_fraction = validation_fraction

    def _build(self, in_dim):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not installed")
        if self.arch == "cnn":
            return TorchCNN1DNet(in_dim)
        if self.arch == "cnngru":
            return TorchCNNGRUNet(in_dim)
        return TorchMLPNet(in_dim)

    def _train_epochs(self, model, X, y, device, epochs, seed):
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()
        ds = TensorDataset(torch.tensor(X), torch.tensor(y))
        gen = torch.Generator()
        gen.manual_seed(int(seed))
        loader = DataLoader(ds, batch_size=min(self.batch_size, len(ds)), shuffle=True, generator=gen)
        for _ in range(int(epochs)):
            model.train()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                opt.step()
        return model

    def fit(self, X, y):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not installed")
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()
        n = len(y)
        if n < 10:
            raise ValueError("Torch validation-based early stopping requires at least 10 calibration samples")

        rng = np.random.default_rng(self.random_state)
        order = rng.permutation(n)
        n_val = max(2, int(round(self.validation_fraction * n)))
        n_val = min(n_val, n - 8)
        tr_idx, va_idx = order[:-n_val], order[-n_val:]
        device = torch.device("cuda" if self.use_gpu and torch.cuda.is_available() else "cpu")
        self.device_ = str(device)


        scaler_inner = StandardScaler().fit(X[tr_idx])
        Xtr = scaler_inner.transform(X[tr_idx]).astype(np.float32)
        Xva = scaler_inner.transform(X[va_idx]).astype(np.float32)
        y_mu = float(np.mean(y[tr_idx]))
        y_sig = float(np.std(y[tr_idx])) if np.std(y[tr_idx]) > 1e-12 else 1.0
        ytr = ((y[tr_idx] - y_mu) / y_sig).astype(np.float32)
        yva = ((y[va_idx] - y_mu) / y_sig).astype(np.float32)

        set_global_seed(self.random_state)
        model = self._build(X.shape[1]).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()
        ds = TensorDataset(torch.tensor(Xtr), torch.tensor(ytr))
        gen = torch.Generator(); gen.manual_seed(int(self.random_state))
        loader = DataLoader(ds, batch_size=min(self.batch_size, len(ds)), shuffle=True, generator=gen)
        Xva_t = torch.tensor(Xva).to(device)
        yva_t = torch.tensor(yva).to(device)

        best_val = float("inf")
        best_epoch = 1
        bad = 0
        for epoch in range(1, self.epochs + 1):
            model.train()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(model(Xva_t), yva_t).item())
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_epoch = epoch
                bad = 0
            else:
                bad += 1
            if bad >= self.patience:
                break

        self.best_epoch_ = int(best_epoch)


        self.scaler_ = StandardScaler().fit(X)
        self.y_mu_ = float(np.mean(y))
        self.y_sig_ = float(np.std(y)) if np.std(y) > 1e-12 else 1.0
        Xs = self.scaler_.transform(X).astype(np.float32)
        ys = ((y - self.y_mu_) / self.y_sig_).astype(np.float32)
        set_global_seed(self.random_state)
        final_model = self._build(X.shape[1]).to(device)
        self.model_ = self._train_epochs(
            final_model, Xs, ys, device, self.best_epoch_,
            stable_seed(self.random_state, self.arch, "full_refit"),
        ).to(device)
        return self

    def predict(self, X):
        Xs = self.scaler_.transform(np.asarray(X, dtype=np.float32)).astype(np.float32)
        device = torch.device(self.device_)
        self.model_.eval()
        with torch.no_grad():
            pred = self.model_(torch.tensor(Xs).to(device)).detach().cpu().numpy().ravel()
        return pred * self.y_sig_ + self.y_mu_


def grids_for(cfg: Config):
    level = cfg.search_level.lower()
    if level == "wide":
        return {
            "krr_alpha": np.logspace(-4, 3, 8),
            "krr_gamma": np.logspace(-4, 0, 7),
            "svr_C": [0.1, 0.3, 1, 3, 10, 30, 100],
            "svr_eps": [0.01, 0.03, 0.05, 0.10, 0.20],
            "svr_gamma": ["scale", 1e-3, 1e-2, 1e-1],
            "lssvm_gamma": np.logspace(-3, 4, 7),
            "lssvm_sigma": np.logspace(-3, 4, 7),
        }
    if level == "fast":
        return {
            "krr_alpha": [1e-2, 1e-1, 1, 10],
            "krr_gamma": [1e-3, 1e-2, 1e-1],
            "svr_C": [0.3, 3, 30],
            "svr_eps": [0.03, 0.10],
            "svr_gamma": ["scale", 1e-2],
            "lssvm_gamma": np.logspace(-1, 3, 4),
            "lssvm_sigma": np.logspace(-1, 3, 4),
        }
    return {
        "krr_alpha": [1e-3, 1e-2, 1e-1, 1, 10, 100],
        "krr_gamma": [1e-4, 1e-3, 1e-2, 1e-1, 1.0],
        "svr_C": [0.3, 1, 3, 10, 30],
        "svr_eps": [0.02, 0.05, 0.10, 0.20],
        "svr_gamma": ["scale", 1e-3, 1e-2, 1e-1],
        "lssvm_gamma": np.logspace(-2, 4, 5),
        "lssvm_sigma": np.logspace(-2, 4, 5),
    }


class StandardizedTargetRegressor(BaseEstimator, RegressorMixin):

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        y = np.asarray(y, dtype=float).ravel()
        self.y_mu_ = float(np.mean(y))
        self.y_sd_ = float(np.std(y)) if np.std(y) > 1e-12 else 1.0
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, (y - self.y_mu_) / self.y_sd_)
        return self

    def predict(self, X):
        z = np.asarray(self.estimator_.predict(X), dtype=float).ravel()
        return z * self.y_sd_ + self.y_mu_


def _candidate_stats(candidate_errors: List[List[float]]):
    means, ses = [], []
    for values in candidate_errors:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            means.append(float("inf")); ses.append(float("inf"))
        else:
            means.append(float(np.mean(arr)))
            ses.append(float(np.std(arr, ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0)
    return np.asarray(means), np.asarray(ses)


def _fold_transform_y(y_raw: np.ndarray, target_method: str, tr: np.ndarray):
    transformer = TargetTransformer(target_method).fit(y_raw[tr])
    return transformer, transformer.transform(y_raw[tr])


def fit_best_pls(X, y_work, cfg: Config, random_state=2026,
                 y_raw: Optional[np.ndarray] = None, target_method: str = "raw"):
    y_work = np.asarray(y_work, dtype=float).ravel()
    y_raw = y_work if y_raw is None else np.asarray(y_raw, dtype=float).ravel()
    max_comp = max(1, min(cfg.max_pls_components, X.shape[1], X.shape[0] - 2))
    candidates = list(range(1, max_comp + 1))
    errors = [[] for _ in candidates]
    for _, _, tr, va in _repeated_kfold_splits(
        len(y_raw), cfg.inner_cv_folds, cfg.inner_cv_repeats, random_state
    ):
        transformer, ytr = _fold_transform_y(y_raw, target_method, tr)
        for i, a in enumerate(candidates):
            try:
                n_comp = max(1, min(a, len(tr) - 2, X.shape[1]))
                mdl = PLSRegression(n_components=n_comp, scale=True).fit(X[tr], ytr)
                pred_raw = transformer.inverse_transform(mdl.predict(X[va]).ravel())
                errors[i].append(safe_rmse(y_raw[va], pred_raw))
            except Exception:
                errors[i].append(float("inf"))
    means, ses = _candidate_stats(errors)
    pos = _select_one_se(means, ses, [float(a) for a in candidates], cfg.use_one_standard_error_rule)
    best_comp = int(candidates[pos])
    model = PLSRegression(n_components=best_comp, scale=True).fit(X, y_work)
    model.best_n_components_ = best_comp
    model.inner_cv_rmse_original_scale_ = float(means[pos])
    return model


def fit_best_ridge(X, y_work, cfg: Config, random_state=2026,
                   y_raw: Optional[np.ndarray] = None, target_method: str = "raw"):
    y_work = np.asarray(y_work, dtype=float).ravel()
    y_raw = y_work if y_raw is None else np.asarray(y_raw, dtype=float).ravel()
    alphas = np.logspace(-2, 6, 17)
    errors = [[] for _ in alphas]
    for _, _, tr, va in _repeated_kfold_splits(
        len(y_raw), cfg.inner_cv_folds, cfg.inner_cv_repeats, random_state
    ):
        transformer, ytr = _fold_transform_y(y_raw, target_method, tr)
        for i, alpha in enumerate(alphas):
            try:
                mdl = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=float(alpha)))])
                mdl.fit(X[tr], ytr)
                pred_raw = transformer.inverse_transform(mdl.predict(X[va]))
                errors[i].append(safe_rmse(y_raw[va], pred_raw))
            except Exception:
                errors[i].append(float("inf"))
    means, ses = _candidate_stats(errors)

    complexity = [float(1.0 / a) for a in alphas]
    pos = _select_one_se(means, ses, complexity, cfg.use_one_standard_error_rule)
    return Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=float(alphas[pos])))]) .fit(X, y_work)


def fit_best_lasso(X, y_work, cfg: Config, random_state=2026,
                   y_raw: Optional[np.ndarray] = None, target_method: str = "raw"):

    y_work = np.asarray(y_work, dtype=float).ravel()
    y_raw = y_work if y_raw is None else np.asarray(y_raw, dtype=float).ravel()
    alphas = np.logspace(-3, 2, 36)
    errors = [[] for _ in alphas]
    for _, _, tr, va in _repeated_kfold_splits(
        len(y_raw), cfg.inner_cv_folds, cfg.inner_cv_repeats, random_state
    ):
        transformer, ytr = _fold_transform_y(y_raw, target_method, tr)
        for i, alpha in enumerate(alphas):
            try:
                mdl = Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", Lasso(alpha=float(alpha), max_iter=20000, random_state=random_state)),
                ]).fit(X[tr], ytr)
                pred_raw = transformer.inverse_transform(mdl.predict(X[va]))
                errors[i].append(safe_rmse(y_raw[va], pred_raw))
            except Exception:
                errors[i].append(float("inf"))
    means, ses = _candidate_stats(errors)
    pos = _select_one_se(means, ses, [float(1.0 / a) for a in alphas],
                         cfg.use_one_standard_error_rule)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", Lasso(alpha=float(alphas[pos]), max_iter=20000, random_state=random_state)),
    ]).fit(X, y_work)


def fit_best_elasticnet(X, y_work, cfg: Config, random_state=2026,
                        y_raw: Optional[np.ndarray] = None, target_method: str = "raw"):

    y_work = np.asarray(y_work, dtype=float).ravel()
    y_raw = y_work if y_raw is None else np.asarray(y_raw, dtype=float).ravel()
    l1_values = [0.10, 0.25, 0.50, 0.75, 0.90]
    alphas = np.logspace(-3, 2, 36)
    candidates = [(float(l1), float(a)) for l1 in l1_values for a in alphas]
    errors = [[] for _ in candidates]
    for _, _, tr, va in _repeated_kfold_splits(
        len(y_raw), cfg.inner_cv_folds, cfg.inner_cv_repeats, random_state
    ):
        transformer, ytr = _fold_transform_y(y_raw, target_method, tr)
        for i, (l1, alpha) in enumerate(candidates):
            try:
                mdl = Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=20000,
                                          random_state=random_state)),
                ]).fit(X[tr], ytr)
                pred_raw = transformer.inverse_transform(mdl.predict(X[va]))
                errors[i].append(safe_rmse(y_raw[va], pred_raw))
            except Exception:
                errors[i].append(float("inf"))
    means, ses = _candidate_stats(errors)
    complexity = [float((1.0 / a) + 0.01 * (1.0 - l1)) for l1, a in candidates]
    pos = _select_one_se(means, ses, complexity, cfg.use_one_standard_error_rule)
    l1, alpha = candidates[pos]
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=20000,
                              random_state=random_state)),
    ]).fit(X, y_work)


def fit_best_krr(X, y_work, cfg: Config, random_state=2026,
                 y_raw: Optional[np.ndarray] = None, target_method: str = "raw"):
    y_work = np.asarray(y_work, dtype=float).ravel()
    y_raw = y_work if y_raw is None else np.asarray(y_raw, dtype=float).ravel()
    grid = grids_for(cfg)
    candidates = [(float(a), float(g)) for a in grid["krr_alpha"] for g in grid["krr_gamma"]]
    errors = [[] for _ in candidates]
    for _, _, tr, va in _repeated_kfold_splits(
        len(y_raw), cfg.inner_cv_folds, cfg.inner_cv_repeats, random_state
    ):
        transformer, ytr_work = _fold_transform_y(y_raw, target_method, tr)
        for i, (alpha, gamma) in enumerate(candidates):
            try:
                base = Pipeline([("scaler", StandardScaler()),
                                 ("model", KernelRidge(kernel="rbf", alpha=alpha, gamma=gamma))])
                mdl = StandardizedTargetRegressor(base).fit(X[tr], ytr_work)
                pred_raw = transformer.inverse_transform(mdl.predict(X[va]))
                errors[i].append(safe_rmse(y_raw[va], pred_raw))
            except Exception:
                errors[i].append(float("inf"))
    means, ses = _candidate_stats(errors)
    complexity = [float((1.0 / max(a, 1e-12)) * g) for a, g in candidates]
    pos = _select_one_se(means, ses, complexity, cfg.use_one_standard_error_rule)
    alpha, gamma = candidates[pos]
    base = Pipeline([("scaler", StandardScaler()),
                     ("model", KernelRidge(kernel="rbf", alpha=alpha, gamma=gamma))])
    return StandardizedTargetRegressor(base).fit(X, y_work)


def fit_best_svr(X, y_work, cfg: Config, kernel="rbf", random_state=2026,
                 y_raw: Optional[np.ndarray] = None, target_method: str = "raw"):
    y_work = np.asarray(y_work, dtype=float).ravel()
    y_raw = y_work if y_raw is None else np.asarray(y_raw, dtype=float).ravel()
    grid = grids_for(cfg)
    if kernel == "rbf":
        candidates = [(float(c), float(e), g) for c in grid["svr_C"]
                      for e in grid["svr_eps"] for g in grid["svr_gamma"]]
    else:
        candidates = [(float(c), float(e), "auto") for c in grid["svr_C"] for e in grid["svr_eps"]]
    errors = [[] for _ in candidates]
    for _, _, tr, va in _repeated_kfold_splits(
        len(y_raw), cfg.inner_cv_folds, cfg.inner_cv_repeats, random_state
    ):
        transformer, ytr_work = _fold_transform_y(y_raw, target_method, tr)
        for i, (cval, eps, gamma) in enumerate(candidates):
            try:
                base = Pipeline([("scaler", StandardScaler()),
                                 ("model", SVR(kernel=kernel, C=cval, epsilon=eps, gamma=gamma))])
                mdl = StandardizedTargetRegressor(base).fit(X[tr], ytr_work)
                pred_raw = transformer.inverse_transform(mdl.predict(X[va]))
                errors[i].append(safe_rmse(y_raw[va], pred_raw))
            except Exception:
                errors[i].append(float("inf"))
    means, ses = _candidate_stats(errors)
    complexity = []
    for cval, eps, gamma in candidates:
        gamma_num = 0.01 if isinstance(gamma, str) else float(gamma)
        complexity.append(float(cval * gamma_num / max(eps, 1e-6)))
    pos = _select_one_se(means, ses, complexity, cfg.use_one_standard_error_rule)
    cval, eps, gamma = candidates[pos]
    base = Pipeline([("scaler", StandardScaler()),
                     ("model", SVR(kernel=kernel, C=cval, epsilon=eps, gamma=gamma))])
    return StandardizedTargetRegressor(base).fit(X, y_work)


def fit_best_pcr(X, y_work, cfg: Config, random_state=2026,
                 y_raw: Optional[np.ndarray] = None, target_method: str = "raw"):
    y_work = np.asarray(y_work, dtype=float).ravel()
    y_raw = y_work if y_raw is None else np.asarray(y_raw, dtype=float).ravel()
    max_comp = max(1, min(cfg.max_pcr_components, X.shape[1], X.shape[0] - 2))
    candidates = list(range(1, max_comp + 1))
    errors = [[] for _ in candidates]
    for _, _, tr, va in _repeated_kfold_splits(
        len(y_raw), cfg.inner_cv_folds, cfg.inner_cv_repeats, random_state
    ):
        transformer, ytr = _fold_transform_y(y_raw, target_method, tr)
        for i, a in enumerate(candidates):
            try:
                mdl = Pipeline([
                    ("scaler", StandardScaler()),
                    ("pca", PCA(n_components=min(a, len(tr) - 1, X.shape[1]), random_state=random_state)),
                    ("ridge", Ridge(alpha=1.0)),
                ]).fit(X[tr], ytr)
                pred_raw = transformer.inverse_transform(mdl.predict(X[va]))
                errors[i].append(safe_rmse(y_raw[va], pred_raw))
            except Exception:
                errors[i].append(float("inf"))
    means, ses = _candidate_stats(errors)
    pos = _select_one_se(means, ses, [float(a) for a in candidates], cfg.use_one_standard_error_rule)
    a = int(candidates[pos])
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=a, random_state=random_state)),
        ("ridge", Ridge(alpha=1.0)),
    ]).fit(X, y_work)


def fit_best_knn(X, y_work, cfg: Config, random_state=2026,
                 y_raw: Optional[np.ndarray] = None, target_method: str = "raw"):
    y_work = np.asarray(y_work, dtype=float).ravel()
    y_raw = y_work if y_raw is None else np.asarray(y_raw, dtype=float).ravel()
    max_pca = max(2, min(15, X.shape[1], X.shape[0] - 2))
    pca_candidates = sorted(set([x for x in (3, 5, 8, 10, 15) if x <= max_pca] + [max_pca]))
    k_candidates = [2, 3, 5, 7, 9]
    candidates = [(pc, k) for pc in pca_candidates for k in k_candidates]
    errors = [[] for _ in candidates]
    for _, _, tr, va in _repeated_kfold_splits(
        len(y_raw), cfg.inner_cv_folds, 1, random_state
    ):
        transformer, ytr = _fold_transform_y(y_raw, target_method, tr)
        for i, (pc, k) in enumerate(candidates):
            if k >= len(tr):
                errors[i].append(float("inf")); continue
            try:
                mdl = Pipeline([
                    ("scaler", StandardScaler()),
                    ("pca", PCA(n_components=min(pc, len(tr) - 1, X.shape[1]), random_state=random_state)),
                    ("model", KNeighborsRegressor(n_neighbors=k, weights="distance", p=2)),
                ]).fit(X[tr], ytr)
                pred_raw = transformer.inverse_transform(mdl.predict(X[va]))
                errors[i].append(safe_rmse(y_raw[va], pred_raw))
            except Exception:
                errors[i].append(float("inf"))
    means, ses = _candidate_stats(errors)
    complexity = [float(pc / max(k, 1)) for pc, k in candidates]
    pos = _select_one_se(means, ses, complexity, cfg.use_one_standard_error_rule)
    pc, k = candidates[pos]
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=pc, random_state=random_state)),
        ("model", KNeighborsRegressor(n_neighbors=k, weights="distance", p=2)),
    ]).fit(X, y_work)


def make_adaboost_regressor(random_state: int):

    tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=2, random_state=random_state)
    try:
        return AdaBoostRegressor(
            estimator=tree,
            n_estimators=200,
            learning_rate=0.05,
            random_state=random_state,
        )
    except TypeError:
        return AdaBoostRegressor(
            base_estimator=tree,
            n_estimators=200,
            learning_rate=0.05,
            random_state=random_state,
        )

def build_model(name: str, cfg: Config, random_state: int):
    upper = name.upper()
    if upper == "PLS":
        return None
    if upper == "RIDGE":
        return None
    if upper == "LASSO":
        return None
    if upper == "ELASTICNET":
        return None
    if upper == "BAYESIANRIDGE":
        return Pipeline([("scaler", StandardScaler()), ("model", BayesianRidge())])
    if upper == "HUBER":
        return Pipeline([("scaler", StandardScaler()), ("model", HuberRegressor(max_iter=2000, alpha=0.1, epsilon=1.35))])
    if upper == "PCR":
        return None
    if upper == "RBF-LS-SVM":
        grid = grids_for(cfg)
        return RBFLSSVMRegressor(gammas=grid["lssvm_gamma"], sigmas=grid["lssvm_sigma"], cv=5, random_state=random_state)
    if upper == "RANDOMFOREST":
        return RandomForestRegressor(n_estimators=500, max_depth=6, min_samples_leaf=3, max_features="sqrt", random_state=random_state, n_jobs=1)
    if upper == "EXTRATREES" or upper == "BAGGEDTREES-OPT":
        return ExtraTreesRegressor(n_estimators=500, max_depth=7, min_samples_leaf=3, max_features="sqrt", random_state=random_state, n_jobs=1)
    if upper == "GRADIENTBOOSTING" or upper == "LSBOOST":
        return GradientBoostingRegressor(n_estimators=250, learning_rate=0.03, max_depth=2, min_samples_leaf=4, subsample=0.85, random_state=random_state)
    if upper == "HISTGBR":
        if not HISTGBR_AVAILABLE:
            raise RuntimeError(
                "HistGradientBoostingRegressor is unavailable in the current scikit-learn environment. "
                "Silent fallback to GradientBoostingRegressor is disabled to prevent a mismatch between the algorithm label and the fitted model."
            )
        return HistGradientBoostingRegressor(
            max_iter=350,
            learning_rate=0.04,
            l2_regularization=0.1,
            random_state=random_state,
        )
    if upper == "ADABOOST":
        return make_adaboost_regressor(random_state)
    if upper == "KNN":
        return None
    if upper == "ANN-MLP" or upper == "ANN":
        return Pipeline([("scaler", StandardScaler()), ("model", MLPRegressor(hidden_layer_sizes=(32, 16), activation="relu", alpha=1e-2, learning_rate_init=5e-4, max_iter=1500, early_stopping=True, validation_fraction=0.20, n_iter_no_change=30, random_state=random_state))])
    if upper == "ELM":
        return ELMRegressor(cv=5, random_state=random_state)
    if upper == "GPR-SE":
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=3.0, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=0.05, noise_level_bounds=(1e-6, 1e1))
        return Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=10, random_state=random_state)),
            ("model", GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, n_restarts_optimizer=1, random_state=random_state)),
        ])
    if upper == "GPR-MATERN32":
        kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=3.0, length_scale_bounds=(1e-2, 1e3), nu=1.5) + WhiteKernel(noise_level=0.05, noise_level_bounds=(1e-6, 1e1))
        return Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=10, random_state=random_state)),
            ("model", GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, n_restarts_optimizer=1, random_state=random_state)),
        ])
    if upper == "PLS-LSSVM":
        return PLSLSSVMRegressor(max_components=cfg.max_pls_components, cv=3, random_state=random_state, gammas=grids_for(cfg)["lssvm_gamma"], sigmas=grids_for(cfg)["lssvm_sigma"])
    if upper == "TORCHMLP":
        return TorchRegressor("mlp", cfg.deep_epochs, cfg.deep_patience, cfg.deep_batch_size, cfg.deep_lr, cfg.deep_weight_decay, random_state, cfg.use_gpu_for_torch, cfg.deep_validation_fraction)
    if upper == "TORCH1DCNN":
        return TorchRegressor("cnn", cfg.deep_epochs, cfg.deep_patience, cfg.deep_batch_size, cfg.deep_lr, cfg.deep_weight_decay, random_state, cfg.use_gpu_for_torch, cfg.deep_validation_fraction)
    if upper == "TORCHCNNGRU":
        return TorchRegressor("cnngru", cfg.deep_epochs, cfg.deep_patience, cfg.deep_batch_size, cfg.deep_lr, cfg.deep_weight_decay, random_state, cfg.use_gpu_for_torch, cfg.deep_validation_fraction)
    if upper == "XGBOOST" and XGBOOST_AVAILABLE:
        return XGBRegressor(n_estimators=300, max_depth=2, min_child_weight=4, learning_rate=0.03, subsample=0.85, colsample_bytree=0.65, reg_alpha=0.1, reg_lambda=5.0, objective="reg:squarederror", random_state=random_state, n_jobs=1)
    if upper == "LIGHTGBM" and LIGHTGBM_AVAILABLE:
        return LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=7, max_depth=3, min_child_samples=10, subsample=0.85, colsample_bytree=0.65, reg_alpha=0.1, reg_lambda=5.0, random_state=random_state, n_jobs=1, verbose=-1)
    if upper == "CATBOOST" and CATBOOST_AVAILABLE:
        return CatBoostRegressor(iterations=300, learning_rate=0.03, depth=3, l2_leaf_reg=8.0, random_strength=1.0, loss_function="RMSE", random_seed=random_state, verbose=False)
    raise ValueError(f"Unknown or unavailable algorithm: {name}")


def fit_predict_base(
    name: str,
    X_train,
    y_train,
    X_test,
    cfg: Config,
    random_state: int,
    y_train_raw: Optional[np.ndarray] = None,
    target_method: str = "raw",
    stacking_context: Optional[Dict[str, Any]] = None,
):
    upper = name.upper()
    y_train = np.asarray(y_train, dtype=float).ravel()
    y_train_raw = y_train if y_train_raw is None else np.asarray(y_train_raw, dtype=float).ravel()
    if upper == "STACKING":
        return fit_predict_stacking(X_train, y_train, X_test, cfg, random_state,
                                    y_train_raw, target_method, stacking_context)
    if upper == "WEIGHTEDSTACKING":
        return fit_predict_weighted_stacking(X_train, y_train, X_test, cfg, random_state,
                                             y_train_raw, target_method, stacking_context)
    if upper == "SUPERSTACKING":
        return fit_predict_super_stacking(X_train, y_train, X_test, cfg, random_state,
                                          y_train_raw, target_method, stacking_context)
    if upper == "PLS":
        model = fit_best_pls(X_train, y_train, cfg, random_state, y_train_raw, target_method)
    elif upper == "RIDGE":
        model = fit_best_ridge(X_train, y_train, cfg, random_state, y_train_raw, target_method)
    elif upper == "LASSO":
        model = fit_best_lasso(X_train, y_train, cfg, random_state, y_train_raw, target_method)
    elif upper == "ELASTICNET":
        model = fit_best_elasticnet(X_train, y_train, cfg, random_state, y_train_raw, target_method)
    elif upper == "PCR":
        model = fit_best_pcr(X_train, y_train, cfg, random_state, y_train_raw, target_method)
    elif upper == "KNN":
        model = fit_best_knn(X_train, y_train, cfg, random_state, y_train_raw, target_method)
    elif upper == "KRR-RBF":
        model = fit_best_krr(X_train, y_train, cfg, random_state, y_train_raw, target_method)
    elif upper == "SVR-RBF":
        model = fit_best_svr(X_train, y_train, cfg, kernel="rbf", random_state=random_state,
                             y_raw=y_train_raw, target_method=target_method)
    elif upper == "SVR-LINEAR":
        model = fit_best_svr(X_train, y_train, cfg, kernel="linear", random_state=random_state,
                             y_raw=y_train_raw, target_method=target_method)
    else:
        model = build_model(name, cfg, random_state)
        model.fit(X_train, y_train)
    return model.predict(X_train).ravel(), model.predict(X_test).ravel()

def _prepare_nested_stacking_folds(
    stacking_context: Dict[str, Any],
    cfg: Config,
    random_state: int,
    n_splits: int = 5,
):

    X_raw = np.asarray(stacking_context["X_train_raw"], dtype=float)
    y_raw = np.asarray(stacking_context["y_train_raw"], dtype=float).ravel()
    method = str(stacking_context["preprocess_method"])
    wavelengths = np.asarray(stacking_context["wavelengths"], dtype=float)
    target_method = str(stacking_context["target_method"])
    outer_transformer: TargetTransformer = stacking_context["outer_target_transformer"]
    k = max(2, min(int(n_splits), len(y_raw)))
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    prepared = []
    for fold_id, (tr, va) in enumerate(kf.split(X_raw), start=1):
        fold_seed = stable_seed(random_state, "stack_nested_fold", fold_id)
        pre = SpectralPreprocessor(method, wavelengths)
        Xtr = pre.fit_transform(X_raw[tr])
        Xva = pre.transform(X_raw[va])
        Xtr, Xva, _ = remove_bad_variables_train_test(Xtr, Xva)
        if Xtr.shape[1] < 2:
            raise RuntimeError(f"Stacking fold {fold_id} has fewer than two valid variables")

        fold_transformer = TargetTransformer(target_method).fit(y_raw[tr])
        ytr_work = fold_transformer.transform(y_raw[tr])
        count = adaptive_feature_count_from_raw(
            X_raw[tr], y_raw[tr], target_method, method, wavelengths, cfg,
            random_state=fold_seed,
        )
        rank = _feature_rank_pls_corr(Xtr, ytr_work, cfg)
        selected = np.sort(rank[:min(int(count), Xtr.shape[1])]).astype(int)
        prepared.append({
            "tr": np.asarray(tr, dtype=int),
            "va": np.asarray(va, dtype=int),
            "Xtr": Xtr[:, selected],
            "Xva": Xva[:, selected],
            "ytr_work": ytr_work,
            "ytr_raw": y_raw[tr],
            "fold_transformer": fold_transformer,
            "outer_transformer": outer_transformer,
        })
    return prepared


def _oof_full_predictions(
    base_name, X_train, y_train, X_test, cfg: Config, random_state: int,
    y_train_raw: Optional[np.ndarray] = None, target_method: str = "raw",
    prepared_folds: Optional[List[Dict[str, Any]]] = None,
):

    y_train = np.asarray(y_train, dtype=float).ravel()
    y_train_raw = y_train if y_train_raw is None else np.asarray(y_train_raw, dtype=float).ravel()
    oof = np.full(len(y_train), np.nan, dtype=float)

    if prepared_folds is not None:
        for fold_id, fold in enumerate(prepared_folds, start=1):
            fold_seed = stable_seed(random_state, base_name, "oof", fold_id)
            _, pred_va_fold_work = fit_predict_base(
                base_name, fold["Xtr"], fold["ytr_work"], fold["Xva"], cfg, fold_seed,
                y_train_raw=fold["ytr_raw"], target_method=target_method,
                stacking_context=None,
            )
            pred_raw = fold["fold_transformer"].inverse_transform(pred_va_fold_work)
            pred_outer_work = fold["outer_transformer"].transform(pred_raw)
            oof[fold["va"]] = pred_outer_work
    else:
        k = max(2, min(5, len(y_train)))
        kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
        for fold_id, (tr, va) in enumerate(kf.split(X_train), start=1):
            fold_seed = stable_seed(random_state, base_name, "oof", fold_id)
            _, pred_va = fit_predict_base(
                base_name, X_train[tr], y_train[tr], X_train[va], cfg, fold_seed,
                y_train_raw=y_train_raw[tr], target_method=target_method,
                stacking_context=None,
            )
            oof[va] = pred_va

    if np.any(~np.isfinite(oof)):
        raise RuntimeError(f"{base_name} OOF predictions contain non-finite values")
    full_seed = stable_seed(random_state, base_name, "full")
    pred_train_full, pred_test_full = fit_predict_base(
        base_name, X_train, y_train, X_test, cfg, full_seed,
        y_train_raw=y_train_raw, target_method=target_method,
        stacking_context=None,
    )
    return oof, pred_train_full, pred_test_full


def _collect_oof_bases(
    base_names, X_train, y_train, X_test, cfg: Config, random_state: int,
    y_train_raw: Optional[np.ndarray] = None, target_method: str = "raw",
    stacking_context: Optional[Dict[str, Any]] = None,
):

    y_train_raw = np.asarray(y_train if y_train_raw is None else y_train_raw, dtype=float).ravel()
    prepared_folds = None
    if cfg.strict_stacking_nested_preprocess:
        if stacking_context is None:
            raise RuntimeError("Strict stacking requires raw calibration context.")
        prepared_folds = _prepare_nested_stacking_folds(
            stacking_context, cfg, stable_seed(random_state, "stack_folds")
        )

    oof_list, train_list, test_list = [], [], []
    for idx, b in enumerate(base_names):
        seed = stable_seed(random_state, "stack_base", idx, b)
        try:
            oof, ptr, pte = _oof_full_predictions(
                b, X_train, y_train, X_test, cfg, seed,
                y_train_raw=y_train_raw, target_method=target_method,
                prepared_folds=prepared_folds,
            )
        except Exception as exc:
            raise RuntimeError(f"Stacking base learner {b} failed: {exc}") from exc
        oof_list.append(oof)
        train_list.append(ptr)
        test_list.append(pte)
    return np.column_stack(oof_list), np.column_stack(train_list), np.column_stack(test_list)


def fit_predict_stacking(
    X_train, y_train, X_test, cfg: Config, random_state: int,
    y_train_raw: Optional[np.ndarray] = None, target_method: str = "raw",
    stacking_context: Optional[Dict[str, Any]] = None,
):
    base_names = ["PLS", "RBF-LS-SVM", "SVR-RBF", "KRR-RBF", "Ridge", "ExtraTrees"]
    Zoof, Ztr, Zte = _collect_oof_bases(
        base_names, X_train, y_train, X_test, cfg, random_state,
        y_train_raw, target_method, stacking_context,
    )
    meta = Ridge(alpha=1.0).fit(Zoof, y_train)
    return meta.predict(Ztr), meta.predict(Zte)


def fit_predict_weighted_stacking(
    X_train, y_train, X_test, cfg: Config, random_state: int,
    y_train_raw: Optional[np.ndarray] = None, target_method: str = "raw",
    stacking_context: Optional[Dict[str, Any]] = None,
):
    base_names = ["PLS", "RBF-LS-SVM", "KRR-RBF", "Ridge", "ExtraTrees"]
    Zoof, Ztr, Zte = _collect_oof_bases(
        base_names, X_train, y_train, X_test, cfg, random_state,
        y_train_raw, target_method, stacking_context,
    )
    rmses = np.asarray([max(complete_oof_rmse(y_train, Zoof[:, j]), 1e-12)
                       for j in range(Zoof.shape[1])])
    if not np.all(np.isfinite(rmses)):
        raise RuntimeError("WeightedStacking contains invalid OOF RMSE values")
    weights = 1.0 / np.square(rmses)
    weights = weights / weights.sum()
    return Ztr @ weights, Zte @ weights


def fit_predict_super_stacking(
    X_train, y_train, X_test, cfg: Config, random_state: int,
    y_train_raw: Optional[np.ndarray] = None, target_method: str = "raw",
    stacking_context: Optional[Dict[str, Any]] = None,
):
    base_names = ["PLS", "RBF-LS-SVM", "KRR-RBF", "SVR-RBF", "Ridge", "PCR", "ELM", "ExtraTrees"]
    if XGBOOST_AVAILABLE:
        base_names.append("XGBoost")
    if LIGHTGBM_AVAILABLE:
        base_names.append("LightGBM")
    Zoof, Ztr, Zte = _collect_oof_bases(
        base_names, X_train, y_train, X_test, cfg, random_state,
        y_train_raw, target_method, stacking_context,
    )
    meta = RidgeCV(alphas=np.logspace(-2, 4, 15)).fit(Zoof, y_train)
    return meta.predict(Ztr), meta.predict(Zte)


def _require_complete_finite_prediction(values: np.ndarray, expected_length: int, label: str) -> np.ndarray:

    arr = np.asarray(values, dtype=float).ravel()
    if arr.size != int(expected_length):
        raise RuntimeError(
            f"{label} length mismatch: expected {expected_length}, got {arr.size}."
        )
    bad = np.flatnonzero(~np.isfinite(arr))
    if bad.size:
        preview = ",".join(map(str, bad[:10].tolist()))
        raise RuntimeError(
            f"{label} contains {bad.size} non-finite values at indices [{preview}]. "
            "The complete configuration is rejected; partial metric calculation is forbidden."
        )
    return arr


def evaluate_one_task(target, algorithm, preprocess, y_transform, data, cfg: Config, random_state: int):
    start = time.time()
    try:
        continuous_algorithms = {"TORCH1DCNN", "TORCHCNNGRU"}
        use_contiguous = str(algorithm).upper() in continuous_algorithms
        X_train = data["X_train_contiguous"] if use_contiguous else data["X_train"]
        X_test = data["X_test_contiguous"] if use_contiguous else data["X_test"]
        used_variable_indices = (
            data["contiguous_variable_indices"] if use_contiguous
            else data["selected_variable_indices"]
        )
        used_wavenumbers = (
            data["contiguous_wavenumbers"] if use_contiguous
            else data["selected_wavenumbers"]
        )
        y_train_work = np.asarray(data["y_train_work"], dtype=float).ravel()
        transformer: TargetTransformer = data["target_transformer"]
        y_train_raw = np.asarray(data["y_train_raw"], dtype=float).ravel()
        y_test_raw = np.asarray(data["y_test_raw"], dtype=float).ravel()
        stacking_context = {
            "X_train_raw": data["X_train_raw"],
            "y_train_raw": y_train_raw,
            "preprocess_method": preprocess,
            "wavelengths": data["wavelengths"],
            "target_method": y_transform,
            "outer_target_transformer": transformer,
        }

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pred_train_work, pred_test_work = fit_predict_base(
                algorithm, X_train, y_train_work, X_test, cfg, random_state,
                y_train_raw=y_train_raw, target_method=y_transform,
                stacking_context=stacking_context,
            )

        warning_messages = []
        for item in caught:
            if issubclass(item.category, (ConvergenceWarning, RuntimeWarning, UserWarning)):
                msg = f"{item.category.__name__}: {item.message}"
                if msg not in warning_messages:
                    warning_messages.append(msg)

        pred_train_work = _require_complete_finite_prediction(
            pred_train_work, len(y_train_work), "Calibration working-scale predictions"
        )
        pred_test_work = _require_complete_finite_prediction(
            pred_test_work, len(y_test_raw), "Prediction-set working-scale predictions"
        )
        pred_train_raw = transformer.inverse_transform(pred_train_work)
        pred_test_raw = transformer.inverse_transform(pred_test_work)
        pred_train_raw = _require_complete_finite_prediction(
            pred_train_raw, len(y_train_raw), "Inverse-transformed calibration predictions"
        )
        pred_test_raw = _require_complete_finite_prediction(
            pred_test_raw, len(y_test_raw), "Inverse-transformed prediction-set predictions"
        )

        rc2 = safe_r2(y_train_raw, pred_train_raw)
        rp2 = safe_r2(y_test_raw, pred_test_raw)
        rmsec = safe_rmse(y_train_raw, pred_train_raw)
        rmsep = safe_rmse(y_test_raw, pred_test_raw)
        mae = safe_mae(y_test_raw, pred_test_raw)
        rpd = float(np.std(y_test_raw, ddof=0) / max(rmsep, 1e-12))
        final_metrics = np.asarray([rc2, rp2, rmsec, rmsep, mae, rpd], dtype=float)
        if not np.all(np.isfinite(final_metrics)):
            raise RuntimeError(f"Final metrics contain non-finite values: {final_metrics}")

        compatibility_split_id = legacy_configuration_split_id(
            cfg, y_transform, preprocess, float(data["train_ratio"]), int(data["repeat_id"])
        )
        res = ModelResult(
            target=target,
            model_name=algorithm,
            preprocess=preprocess,
            y_transform=y_transform,
            train_ratio=float(data["train_ratio"]),
            repeat_id=int(data["repeat_id"]),
            split_id=compatibility_split_id,
            n_original_samples=int(data["n_original_samples"]),
            n_samples=int(data["n_samples"]),
            n_presearch_outliers=int(data["n_presearch_outliers"]),
            n_calibration_raw=int(data["n_calibration_raw"]),
            n_calibration_used=int(data["n_calibration_used"]),
            n_prediction=int(data["n_prediction"]),
            n_calibration_outliers=int(data["n_calibration_outliers"]),
            n_features=int(X_train.shape[1]),
            n_effective_variables=int(data["n_effective_variables"]),
            rc2=rc2,
            rp2=rp2,
            rmsec=rmsec,
            rmsep=rmsep,
            mae=mae,
            rpd=rpd,
            y_train=y_train_raw,
            y_test=y_test_raw,
            pred_train=pred_train_raw,
            pred_test=pred_test_raw,
            train_sample_names=list(data["train_sample_names"]),
            test_sample_names=list(data["test_sample_names"]),
            presearch_outlier_names=list(data["presearch_outlier_names"]),
            calibration_outlier_names=list(data.get("calibration_outlier_names", [])),
            selected_variable_indices=np.asarray(used_variable_indices, dtype=int),
            selected_wavenumbers=np.asarray(used_wavenumbers, dtype=float),
            elapsed_sec=time.time() - start,
            outer_split_id=int(data["split_id"]),
            warning_count=len(warning_messages),
            warning_messages=warning_messages,
        )
        return res, ""
    except Exception as exc:
        return None, str(exc)

def prediction_rows(item: ModelResult):
    rows = []
    for set_type, names, y_true, y_pred in [
        ("Calibration", item.train_sample_names, item.y_train, item.pred_train),
        ("Prediction", item.test_sample_names, item.y_test, item.pred_test),
    ]:
        for name, yt, yp in zip(names, y_true, y_pred):
            err = float(yp - yt)
            rows.append({
                "target": item.target,
                "modelName": item.model_name,
                "preprocess": item.preprocess,
                "yTransform": item.y_transform,
                "setType": set_type,
                "sampleName": str(name),
                "actualValue": float(yt),
                "predictedValue": float(yp),
                "error": err,
                "absError": abs(err),
                "trainRatio": item.train_ratio,
                "repeatId": item.repeat_id,
                "splitId": item.split_id,
                "Rc2": item.rc2,
                "Rp2": item.rp2,
                "RMSEC": item.rmsec,
                "RMSEP": item.rmsep,
                "MAE": item.mae,
                "RPD": item.rpd,
            })
    return rows


def best_by_algorithm(records: List[ModelResult]):

    out = []
    for name in sorted({r.model_name for r in records}):
        sub = [r for r in records if r.model_name == name]
        sub = sorted(sub, key=lambda r: (np.nan_to_num(r.rp2, nan=-np.inf), -r.rmsep), reverse=True)
        if sub:
            out.append(sub[0])
    return sorted(out, key=lambda r: (np.nan_to_num(r.rp2, nan=-np.inf), -r.rmsep), reverse=True)


def algorithm_list(cfg: Config):
    missing = []
    algs = list(cfg.algorithms)

    if cfg.use_xgboost:
        if not XGBOOST_AVAILABLE:
            missing.append("XGBoost")
        else:
            algs.append("XGBoost")
    if cfg.use_lightgbm:
        if not LIGHTGBM_AVAILABLE:
            missing.append("LightGBM")
        else:
            algs.append("LightGBM")
    if cfg.use_catboost:
        if not CATBOOST_AVAILABLE:
            missing.append("CatBoost")
        else:
            algs.append("CatBoost")

    if cfg.enable_deep_learning:
        if not TORCH_AVAILABLE:
            missing.extend(list(cfg.deep_algorithms))
        else:
            algs.extend(list(cfg.deep_algorithms))

    algs = list(dict.fromkeys(algs))
    if any(str(a).upper() == "HISTGBR" for a in algs) and not HISTGBR_AVAILABLE:
        missing.append("HistGBR")

    if missing:
        raise RuntimeError("Algorithms required by the manuscript workflow are unavailable: " + ", ".join(sorted(set(missing))))
    if cfg.require_exact_30_algorithms and len(algs) != 30:
        raise RuntimeError(f"The manuscript workflow requires exactly 30 algorithms; current count is {len(algs)}: {algs}")
    return algs


def validate_factorial_design(cfg: Config, algs: List[str]):
    if len(cfg.preprocess_methods) != 28:
        raise RuntimeError(f"The manuscript workflow requires 28 preprocessing methods; current count is {len(cfg.preprocess_methods)}")
    if len(cfg.target_transforms) != 4:
        raise RuntimeError(f"The manuscript workflow requires 4 response transformations; current count is {len(cfg.target_transforms)}")
    if tuple(cfg.train_ratio_list) != (0.75, 0.80, 0.85):
        raise RuntimeError(f"The manuscript workflow requires calibration proportions 0.75/0.80/0.85; current values are {cfg.train_ratio_list}")
    if cfg.n_repeated_split != 10:
        raise RuntimeError(f"The manuscript workflow requires 10 repeated splits; current value is {cfg.n_repeated_split}")
    if len(algs) != 30:
        raise RuntimeError(f"The manuscript workflow requires 30 algorithms; current count is {len(algs)}")
    if str(cfg.feature_selection_mode).lower() == "fixed_count_nested" and int(cfg.fixed_feature_count) != 158:
        raise RuntimeError(f"The reference workflow requires a fixed count of 158 variables; current value is {cfg.fixed_feature_count}")
    if str(cfg.outer_split_method).lower() != "response_stratified_random":
        raise RuntimeError("The release workflow must use 10 response-stratified random splits.")
    expected = 28 * 4 * 3 * 10 * 30
    actual = len(cfg.preprocess_methods) * len(cfg.target_transforms) * len(cfg.train_ratio_list) * cfg.n_repeated_split * len(algs)
    if actual != expected:
        raise RuntimeError(f"Factorial space mismatch: expected {expected}, observed {actual}")


def prepare_cache_for_split(X0, y0, sample_names0, wavelengths, split: SplitSpec, cfg: Config, population_meta: Dict[str, Any]):

    cache = {}
    raw_train_indices = np.asarray(split.train_idx, dtype=int)
    X_train_raw_all = X0[raw_train_indices]
    y_train_raw_all = y0[raw_train_indices]
    train_names_all = sample_names0[raw_train_indices]
    X_test_raw = X0[split.test_idx]
    y_test_raw = y0[split.test_idx]

    calibration_keep = np.ones(len(y_train_raw_all), dtype=bool)
    if cfg.enable_outlier_filter and cfg.outlier_filter_scope.lower() == "calibration_only":
        out_seed = stable_seed(cfg.random_state, "calibration_outliers", split.split_id)
        calibration_keep = mccv_outlier_keep(
            X_train_raw_all, y_train_raw_all, cfg, random_state=out_seed
        )

        minimum_keep = max(12, int(round(0.85 * len(y_train_raw_all))))
        if int(calibration_keep.sum()) < minimum_keep:
            calibration_keep[:] = True

    X_train_raw = X_train_raw_all[calibration_keep]
    y_train_raw = y_train_raw_all[calibration_keep]
    train_names = train_names_all[calibration_keep]
    calibration_outlier_names = train_names_all[~calibration_keep].astype(str).tolist()

    for y_transform in cfg.target_transforms:
        try:
            transformer = TargetTransformer(y_transform).fit(y_train_raw)
            y_train_work = transformer.transform(y_train_raw)
        except Exception as exc:
            print(f"  Skipping response transformation={y_transform}: {exc}")
            continue

        for method in cfg.preprocess_methods:
            try:
                pre = SpectralPreprocessor(method, wavelengths)
                Xp_train = pre.fit_transform(X_train_raw)
                Xp_test = pre.transform(X_test_raw)

                Xp_train, Xp_test, good_mask = remove_bad_variables_train_test(Xp_train, Xp_test)
                if Xp_train.shape[1] < 2:
                    print(f"  Skipping preprocessing {method}: fewer than 2 valid variables remain in the calibration set")
                    continue

                fs_seed = stable_seed(
                    cfg.random_state, "adaptive_feature_selection",
                    split.split_id, y_transform, method,
                )
                chosen_count = adaptive_feature_count_from_raw(
                    X_train_raw, y_train_raw, y_transform, method, wavelengths, cfg,
                    random_state=fs_seed,
                )
                final_rank = _feature_rank_pls_corr(Xp_train, y_train_work, cfg)
                if chosen_count >= Xp_train.shape[1]:
                    selected = np.arange(Xp_train.shape[1], dtype=int)
                else:
                    selected = np.sort(final_rank[:chosen_count]).astype(int)
                if selected.size == 0:
                    selected = np.arange(Xp_train.shape[1], dtype=int)

                Xs_train = Xp_train[:, selected]
                Xs_test = Xp_test[:, selected]
                contiguous = _select_contiguous_feature_window(
                    Xp_train, y_train_work, min(int(cfg.fixed_feature_count), Xp_train.shape[1]), cfg
                )
                Xc_train = Xp_train[:, contiguous]
                Xc_test = Xp_test[:, contiguous]
                good_original_indices = np.flatnonzero(good_mask)
                selected_original_indices = good_original_indices[selected]
                contiguous_original_indices = good_original_indices[contiguous]
                selected_wavenumbers = np.asarray(wavelengths, dtype=float)[selected_original_indices]
                contiguous_wavenumbers = np.asarray(wavelengths, dtype=float)[contiguous_original_indices]

                cache[(y_transform, method)] = {
                    "X_train": Xs_train,
                    "X_test": Xs_test,
                    "X_train_contiguous": Xc_train,
                    "X_test_contiguous": Xc_test,
                    "X_train_raw": X_train_raw,
                    "X_test_raw": X_test_raw,
                    "wavelengths": np.asarray(wavelengths, dtype=float),
                    "y_train_work": y_train_work,
                    "target_transformer": transformer,
                    "y_train_raw": y_train_raw,
                    "y_test_raw": y_test_raw,
                    "train_sample_names": train_names.astype(str).tolist(),
                    "test_sample_names": sample_names0[split.test_idx].astype(str).tolist(),
                    "train_ratio": split.train_ratio,
                    "repeat_id": split.repeat_id,
                    "split_id": split.split_id,
                    "n_original_samples": int(population_meta["n_original_samples"]),
                    "n_samples": len(y0),
                    "n_presearch_outliers": int(population_meta["n_presearch_outliers"]),
                    "presearch_outlier_names": list(population_meta["presearch_outlier_names"]),
                    "calibration_outlier_names": calibration_outlier_names,
                    "n_calibration_raw": len(raw_train_indices),
                    "n_calibration_used": len(y_train_raw),
                    "n_prediction": len(split.test_idx),
                    "n_calibration_outliers": int((~calibration_keep).sum()),
                    "n_features": int(len(selected)),
                    "n_effective_variables": int(Xp_train.shape[1]),
                    "selected_variable_indices": selected_original_indices,
                    "selected_wavenumbers": selected_wavenumbers,
                    "contiguous_variable_indices": contiguous_original_indices,
                    "contiguous_wavenumbers": contiguous_wavenumbers,
                }
                print(
                    f"  Cache ready: split={split.split_id:02d} | y={y_transform:8s} | "
                    f"preprocessing={method:20s} | valid variables={Xp_train.shape[1]:4d} | "
                    f"selected variables={len(selected):4d} | contiguous window={len(contiguous):4d}"
                )
            except Exception as exc:
                print(f"  Preprocessing failed: split={split.split_id} | y={y_transform} | {method} | {exc}")
    return cache


def run_parallel_tasks(target, tasks, cfg: Config, progress_label: str = ""):
    total = len(tasks)
    print(f"Starting analyte [{target}] {progress_label} tasks; total combinations: {total}")
    start = time.time()

    def _run_one(i, task):
        algorithm, y_transform, method, data, model_seed = task
        if cfg.show_realtime_log:
            print(
                f"[{i+1:6d}/{total:6d}] Start: analyte={target} | algorithm={algorithm} | "
                f"preprocessing={method} | y={y_transform} | ratio={data['train_ratio']:.2f} | "
                f"rep={data['repeat_id']} | split={data['split_id']} | features={data['n_features']}",
                flush=True,
            )
        res, err = evaluate_one_task(target, algorithm, method, y_transform, data, cfg, model_seed)
        if cfg.show_realtime_log:
            if res is not None:
                print(
                    f"[{i+1:6d}/{total:6d}] Done: {algorithm:18s} | {method:20s} | "
                    f"y={y_transform:8s} | Rc2={res.rc2:7.4f} | Rp2={res.rp2:7.4f} | "
                    f"RMSEC={res.rmsec:10.4f} | RMSEP={res.rmsep:10.4f} | "
                    f"RPD={res.rpd:7.3f} | {res.elapsed_sec:6.2f}s",
                    flush=True,
                )
            else:
                msg = err if cfg.show_failed_reason else "failed or skipped"
                print(f"[{i+1:6d}/{total:6d}] Failed: {algorithm} | {method} | y={y_transform} | {msg}", flush=True)
        return res, err


    deep_names = {x.upper() for x in cfg.deep_algorithms}
    serial_classical_names = {"CATBOOST"}

    classical = [
        (i, t) for i, t in enumerate(tasks)
        if str(t[0]).upper() not in deep_names
        and str(t[0]).upper() not in serial_classical_names
    ]
    serial_classical = [
        (i, t) for i, t in enumerate(tasks)
        if str(t[0]).upper() in serial_classical_names
    ]
    deep = [
        (i, t) for i, t in enumerate(tasks)
        if str(t[0]).upper() in deep_names
    ]

    if len(classical) + len(serial_classical) + len(deep) != total:
        raise RuntimeError(
            "Task scheduling groups are incomplete: "
            f"parallel_classical={len(classical)}, "
            f"serial_catboost={len(serial_classical)}, "
            f"serial_torch={len(deep)}, total={total}"
        )

    print(
        f"Task scheduling: parallel classical models {len(classical)} | "
        f"serial CatBoost tasks {len(serial_classical)} | "
        f"serial Torch tasks {len(deep)}"
    )

    outputs = []


    if classical:
        outputs.extend(
            Parallel(
                n_jobs=cfg.n_jobs,
                backend=cfg.parallel_backend,
                verbose=0,
            )(
                delayed(_run_one)(i, task) for i, task in classical
            )
        )


    for i, task in serial_classical:
        outputs.append(_run_one(i, task))


    for i, task in deep:
        outputs.append(_run_one(i, task))

    records = [r for r, _ in outputs if r is not None and np.isfinite(r.rmsep)]
    failures = [err for r, err in outputs if r is None or not np.isfinite(getattr(r, "rmsep", np.nan))]
    print(f"Analyte [{target}] {progress_label} completed: {len(records)}/{total} successful; elapsed {(time.time() - start)/60:.2f} min")
    return records, failures


def run_target(target, X0, y0, sample_names0, wavelengths, cfg: Config, population_meta: Optional[Dict[str, Any]] = None):
    print("=" * 80)
    print(f"Modeling analyte: {target}")
    print(f"Original valid sample count: {len(y0)}")
    if len(y0) < 8 or np.std(y0) < 1e-12:
        print("Skipping: too few samples or insufficient response variation.")
        return [], [], None

    if cfg.outlier_filter_scope.lower() not in {"disabled", "calibration_only"}:
        raise ValueError("outlier_filter_scope must be either 'disabled' or 'calibration_only'.")

    algs = algorithm_list(cfg)
    validate_factorial_design(cfg, algs)


    X_model = X0
    y_model = y0
    sn_model = sample_names0
    if population_meta is None:
        population_meta = {
            "n_original_samples": int(len(y0)),
            "n_presearch_outliers": 0,
            "presearch_outlier_names": [],
        }
    original_n = int(population_meta["n_original_samples"])

    print(
        f"Fixed modeling population before outer splitting: {len(y_model)} / {original_n}; "
        f"predeclared or missing-reference exclusions={population_meta['n_presearch_outliers']}"
    )
    if cfg.enable_outlier_filter and cfg.outlier_filter_scope.lower() == "calibration_only":
        print("Outlier-screening mode: screening is restricted to the outer calibration set; prediction-set samples are not removed.")
    else:
        print("Outlier-screening mode: disabled (recommended setting for the manuscript primary analysis).")

    print(
        f"Algorithms={len(algs)} | preprocessing methods={len(cfg.preprocess_methods)} | "
        f"response transformations={len(cfg.target_transforms)} | calibration proportions={len(cfg.train_ratio_list)} | "
        f"repeated splits={cfg.n_repeated_split}"
    )
    expected_tasks = 28 * 4 * 3 * 10 * 30
    print(f"Expected combination count: {expected_tasks}")

    split_specs = build_shared_outer_splits(X_model, y_model, cfg)
    print(f"Generated and fixed shared outer splits: {len(split_specs)}")
    for s in split_specs:
        print(
            f"  split={s.split_id:02d} | ratio={s.train_ratio:.2f} | rep={s.repeat_id} | "
            f"cal={len(s.train_idx)} | pred={len(s.test_idx)}"
        )

    all_records: List[ModelResult] = []
    all_failures: List[str] = []
    for split in split_specs:
        print("-" * 80)
        print(f"Preparing split={split.split_id:02d} | ratio={split.train_ratio:.2f} | rep={split.repeat_id}")
        cache = prepare_cache_for_split(
            X_model, y_model, sn_model, wavelengths, split, cfg, population_meta
        )
        expected_cache = len(cfg.target_transforms) * len(cfg.preprocess_methods)
        if cfg.require_complete_factorial and len(cache) != expected_cache:
            raise RuntimeError(
                f"Split {split.split_id} cache is incomplete: expected {expected_cache}, observed {len(cache)}"
            )

        tasks = []
        for (y_transform, method), data in cache.items():
            for alg in algs:
                model_seed = stable_seed(
                    cfg.random_state, "model", target, split.split_id, y_transform, method, alg
                )
                tasks.append((alg, y_transform, method, data, model_seed))

        split_records, failures = run_parallel_tasks(
            target, tasks, cfg,
            progress_label=f" split {split.split_id:02d}/{len(split_specs):02d} ",
        )
        all_records.extend(split_records)
        all_failures.extend(failures)

        if cfg.require_complete_factorial and len(split_records) != len(tasks):
            examples = [x for x in failures if x][:5]
            raise RuntimeError(
                f"Split {split.split_id} did not generate all models: {len(split_records)}/{len(tasks)} successful. "
                f"Failure examples: {examples}"
            )

    records = all_records
    if cfg.require_complete_factorial and len(records) != expected_tasks:
        raise RuntimeError(f"Analyte {target} expected {expected_tasks} complete records, observed {len(records)}.")
    if not records:
        print(f"All models failed for analyte [{target}].")
        return [], [], None

    records = sorted(records, key=lambda r: (np.nan_to_num(r.rp2, nan=-np.inf), -r.rmsep), reverse=True)
    best_alg_records = best_by_algorithm(records)
    best = records[0]

    if cfg.show_detail_each_target:
        print(f"\n---------- Analyte [{target}] top {cfg.show_top_n} workflow evaluations ----------")
        print(pd.DataFrame([r.metric_row() for r in records[:cfg.show_top_n]]).to_string(index=False))
        print(f"\n---------- Analyte [{target}] post-selection upper-envelope result by algorithm ----------")
        print(pd.DataFrame([r.metric_row() for r in best_alg_records]).to_string(index=False))

    print(
        f"\nAnalyte [{target}] post-selection internal holdout upper-envelope maximum: algorithm={best.model_name} | preprocessing={best.preprocess} | "
        f"y={best.y_transform} | Ratio={best.train_ratio:.2f} | Rep={best.repeat_id} | "
        f"Rc2={best.rc2:.4f} | Rp2={best.rp2:.4f} | RMSEP={best.rmsep:.4f} | RPD={best.rpd:.3f}\n"
    )
    return records, best_alg_records, best


def _safe_sheet_name(name: str) -> str:

    bad = ['\\', '/', '*', '?', ':', '[', ']']
    for ch in bad:
        name = name.replace(ch, '_')
    return name[:31]


def _write_dataframe_split(writer, df: pd.DataFrame, base_sheet: str, max_rows: int = 1_000_000):

    base_sheet = _safe_sheet_name(base_sheet)
    if df is None or df.empty:
        pd.DataFrame().to_excel(writer, sheet_name=base_sheet, index=False)
        return
    if len(df) <= max_rows:
        df.to_excel(writer, sheet_name=base_sheet, index=False)
        return
    n_parts = int(math.ceil(len(df) / max_rows))
    for i in range(n_parts):
        part = df.iloc[i * max_rows:(i + 1) * max_rows]
        suffix = f"_{i + 1}"
        sheet = _safe_sheet_name(base_sheet[:31 - len(suffix)] + suffix)
        part.to_excel(writer, sheet_name=sheet, index=False)


def _split_manifest(all_records: List[ModelResult]) -> pd.DataFrame:
    seen = {}
    for r in all_records:
        key = (r.target, r.outer_split_id)
        if key not in seen:
            seen[key] = {
                "target": r.target,
                "outerSplitId": r.outer_split_id,
                "trainRatio": r.train_ratio,
                "repeatId": r.repeat_id,
                "calibrationSamples": ",".join(r.train_sample_names),
                "predictionSamples": ",".join(r.test_sample_names),
                "nCalibration": len(r.train_sample_names),
                "nPrediction": len(r.test_sample_names),
            }
    return pd.DataFrame(seen.values())


def _outlier_manifest(all_records: List[ModelResult]) -> pd.DataFrame:
    seen = {}
    for r in all_records:
        key = (r.target, r.outer_split_id)
        if key not in seen:
            seen[key] = {
                "target": r.target,
                "outerSplitId": r.outer_split_id,
                "trainRatio": r.train_ratio,
                "repeatId": r.repeat_id,
                "nOriginalSamples": r.n_original_samples,
                "nOuterPopulationSamples": r.n_samples,
                "nPresearchExcluded": r.n_presearch_outliers,
                "presearchExcludedNames": ",".join(r.presearch_outlier_names),
                "nCalibrationRaw": r.n_calibration_raw,
                "nCalibrationUsed": r.n_calibration_used,
                "nCalibrationExcluded": r.n_calibration_outliers,
                "calibrationExcludedNames": ",".join(r.calibration_outlier_names),
                "nPrediction": r.n_prediction,
            }
    return pd.DataFrame(seen.values())


def _selected_variable_manifest(all_records: List[ModelResult]) -> pd.DataFrame:

    seen = {}
    continuous_names = {"TORCH1DCNN", "TORCHCNNGRU"}
    for r in all_records:
        feature_path = "contiguous158" if r.model_name.upper() in continuous_names else "ranked158"
        key = (r.target, r.outer_split_id, r.y_transform, r.preprocess, feature_path)
        if key not in seen:
            seen[key] = {
                "target": r.target,
                "outerSplitId": r.outer_split_id,
                "splitId": r.split_id,
                "trainRatio": r.train_ratio,
                "repeatId": r.repeat_id,
                "yTransform": r.y_transform,
                "preprocess": r.preprocess,
                "featurePath": feature_path,
                "nFeatures": r.n_features,
                "selectedVariableIndices": ",".join(map(str, r.selected_variable_indices.tolist())),
                "selectedRamanShifts_cm-1": ",".join(f"{x:.6f}" for x in r.selected_wavenumbers.tolist()),
            }
    return pd.DataFrame(seen.values())


METRIC_OUTPUT_COLUMNS = [
    "target", "modelName", "preprocess", "yTransform", "trainRatio",
    "repeatId", "splitId", "Rc2", "Rp2", "RMSEC", "RMSEP", "MAE",
    "RPD", "nSamples", "nFeatures", "nEffectiveVariables", "elapsedSec",
]

PREDICTION_OUTPUT_COLUMNS = [
    "target", "modelName", "preprocess", "yTransform", "setType",
    "sampleName", "actualValue", "predictedValue", "error", "absError",
    "trainRatio", "repeatId", "splitId", "Rc2", "Rp2", "RMSEC",
    "RMSEP", "MAE", "RPD",
]


def export_excel(best_summary, all_records, best_by_alg_records, cfg: Config,
                 algs: Optional[List[str]] = None,
                 provenance: Optional[Dict[str, Any]] = None,
                 exclusion_audit_rows: Optional[List[Dict[str, Any]]] = None):

    best_df = pd.DataFrame([r.metric_row() for r in best_summary]).reindex(columns=METRIC_OUTPUT_COLUMNS)
    all_df = pd.DataFrame([r.metric_row() for r in all_records]).reindex(columns=METRIC_OUTPUT_COLUMNS)
    alg_df = pd.DataFrame([r.metric_row() for r in best_by_alg_records]).reindex(columns=METRIC_OUTPUT_COLUMNS)

    pred_rows = []
    if cfg.save_prediction_details:
        for r in best_by_alg_records:
            pred_rows.extend(prediction_rows(r))
    pred_df = pd.DataFrame(pred_rows).reindex(columns=PREDICTION_OUTPUT_COLUMNS)

    target_order = {r.target: i for i, r in enumerate(best_summary)}
    if not all_df.empty:
        all_df["_targetOrder"] = all_df["target"].map(target_order).fillna(len(target_order))
        all_df = all_df.sort_values(["_targetOrder", "Rp2", "RMSEP"],
                                    ascending=[True, False, True]).drop(columns="_targetOrder")
    if not alg_df.empty:
        alg_df["_targetOrder"] = alg_df["target"].map(target_order).fillna(len(target_order))
        alg_df = alg_df.sort_values(["_targetOrder", "Rp2", "RMSEP"],
                                    ascending=[True, False, True]).drop(columns="_targetOrder")
    if not pred_df.empty:
        pred_df["_targetOrder"] = pred_df["target"].map(target_order).fillna(len(target_order))
        pred_df = pred_df.sort_values(
            ["_targetOrder", "modelName", "setType", "sampleName"]
        ).drop(columns="_targetOrder")


    if list(best_df.columns) != METRIC_OUTPUT_COLUMNS:
        raise RuntimeError("BestSummary column structure does not match the reference workbook.")
    if list(alg_df.columns) != METRIC_OUTPUT_COLUMNS:
        raise RuntimeError("BestByTargetAlgorithm column structure does not match the reference workbook.")
    if list(all_df.columns) != METRIC_OUTPUT_COLUMNS:
        raise RuntimeError("AllRunMetrics column structure does not match the reference workbook.")
    if list(pred_df.columns) != PREDICTION_OUTPUT_COLUMNS:
        raise RuntimeError("PredictionDetails column structure does not match the reference workbook.")

    output_path = Path(cfg.output_excel)
    csv_dir = output_path.with_suffix("")
    if cfg.save_auxiliary_audit_csv:
        csv_dir.mkdir(parents=True, exist_ok=True)
        split_df = _split_manifest(all_records)
        outlier_df = _outlier_manifest(all_records)
        variable_df = _selected_variable_manifest(all_records)
        warning_rows = []
        for r in all_records:
            if r.warning_count:
                warning_rows.append({
                    "target": r.target,
                    "modelName": r.model_name,
                    "preprocess": r.preprocess,
                    "yTransform": r.y_transform,
                    "trainRatio": r.train_ratio,
                    "repeatId": r.repeat_id,
                    "splitId": r.split_id,
                    "outerSplitId": r.outer_split_id,
                    "warningCount": r.warning_count,
                    "warningMessages": " | ".join(r.warning_messages),
                })
        sig_hash, sig_payload = run_signature(cfg, algs, provenance)
        metadata_df = pd.DataFrame([
            {"key": "pipelineVersion", "value": cfg.pipeline_version},
            {"key": "runSignature", "value": sig_hash},
            {"key": "selectionSemantics",
             "value": "BestSummary and BestByTargetAlgorithm are post-selection internal-hold-out upper-envelope summaries, not external validation."},
            {"key": "scientificConfig", "value": json.dumps(sig_payload, ensure_ascii=False, sort_keys=True)},
            {"key": "provenanceLock", "value": json.dumps(provenance, ensure_ascii=False, sort_keys=True)},
            {"key": "allRunMetricsRows", "value": len(all_df)},
        ])
        best_df.to_csv(csv_dir / "BestSummary.csv", index=False, encoding="utf-8-sig")
        alg_df.to_csv(csv_dir / "BestByTargetAlgorithm.csv", index=False, encoding="utf-8-sig")
        all_df.to_csv(csv_dir / "AllRunMetrics.csv", index=False, encoding="utf-8-sig")
        pred_df.to_csv(csv_dir / "PredictionDetails.csv", index=False, encoding="utf-8-sig")
        split_df.to_csv(csv_dir / "SplitManifest.csv", index=False, encoding="utf-8-sig")
        outlier_df.to_csv(csv_dir / "OutlierManifest.csv", index=False, encoding="utf-8-sig")
        variable_df.to_csv(csv_dir / "SelectedVariableManifest.csv", index=False, encoding="utf-8-sig")
        metadata_df.to_csv(csv_dir / "RunMetadata.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(warning_rows).to_csv(csv_dir / "RunWarnings.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(exclusion_audit_rows or []).to_csv(
            csv_dir / "PredeclaredExclusionAudit.csv", index=False, encoding="utf-8-sig"
        )


    with pd.ExcelWriter(cfg.output_excel, engine="openpyxl") as writer:
        _write_dataframe_split(writer, best_df, "BestSummary")
        _write_dataframe_split(writer, alg_df, "BestByTargetAlgorithm")
        _write_dataframe_split(writer, all_df, "AllRunMetrics")
        _write_dataframe_split(writer, pred_df, "PredictionDetails")

    return best_df, alg_df, all_df, pred_df


def _target_key(name: Any) -> str:

    return str(name).strip()


def load_checkpoint(cfg: Config, algs: Optional[List[str]] = None,
                    provenance: Optional[Dict[str, Any]] = None):
    ckpt_path = Path(cfg.checkpoint_file)
    if not cfg.enable_resume or not ckpt_path.exists():
        return [], [], [], set()

    expected_hash, expected_payload = run_signature(cfg, algs, provenance)
    try:
        with open(ckpt_path, "rb") as f:
            ckpt = pickle.load(f)

        saved_hash = ckpt.get("run_signature")
        if saved_hash != expected_hash:
            raise RuntimeError(
                "Checkpoint scientific configuration differs from the current run; mixing two experimental protocols is not allowed.\n"
                f"saved={saved_hash}\ncurrent={expected_hash}\n"
                f"currentConfig={json.dumps(expected_payload, ensure_ascii=False, sort_keys=True)}"
            )

        all_records = ckpt.get("all_records", [])
        all_best_by_alg = ckpt.get("all_best_by_alg", [])
        best_summary = ckpt.get("best_summary", [])
        completed_targets = set(_target_key(x) for x in ckpt.get("completed_targets", []))

        print("=" * 80)
        print(f"Compatible checkpoint detected: {ckpt_path}")
        print(f"Checkpoint save time: {ckpt.get('save_time', 'unknown')}")
        print(f"Run signature: {saved_hash}")
        print(f"Completed analyte count: {len(completed_targets)}")
        if completed_targets:
            print("Completed analytes:")
            for name in sorted(completed_targets):
                print(f"  - {name}")
        print("These analytes will be skipped and unfinished analytes will continue.")
        print("=" * 80)
        return all_records, all_best_by_alg, best_summary, completed_targets

    except Exception as exc:
        print("=" * 80)
        print(f"Failed to read checkpoint file: {exc}")
        print("To prevent mixing different experimental protocols, this release does not silently ignore an incompatible checkpoint.")
        print("After verification, delete or rename the old checkpoint and rerun.")
        print("=" * 80)
        raise


def save_checkpoint(cfg: Config, all_records, all_best_by_alg, best_summary, completed_targets,
                    algs: Optional[List[str]] = None,
                    provenance: Optional[Dict[str, Any]] = None):
    if not cfg.enable_resume:
        return

    ckpt_path = Path(cfg.checkpoint_file)
    tmp_path = ckpt_path.with_suffix(ckpt_path.suffix + ".tmp")
    sig_hash, sig_payload = run_signature(cfg, algs, provenance)
    ckpt = {
        "pipeline_version": cfg.pipeline_version,
        "run_signature": sig_hash,
        "scientific_config": sig_payload,
        "provenance_lock": provenance,
        "all_records": all_records,
        "all_best_by_alg": all_best_by_alg,
        "best_summary": best_summary,
        "completed_targets": sorted([_target_key(x) for x in completed_targets]),
        "save_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    try:
        with open(tmp_path, "wb") as f:
            pickle.dump(ckpt, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, ckpt_path)
        print(f"Checkpoint saved: {ckpt_path}")
    except Exception as exc:
        print(f"Checkpoint save failed: {exc}")
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        raise


def remove_checkpoint_if_finished(cfg: Config):

    return


def print_environment(cfg: Config):
    import sklearn
    print("========== Runtime environment ==========")
    print(f"Python PID: {os.getpid()}")
    print(f"scikit-learn: {sklearn.__version__}")
    print(f"HistGBR available: {HISTGBR_AVAILABLE}")
    print(f"XGBoost available: {XGBOOST_AVAILABLE} | LightGBM available: {LIGHTGBM_AVAILABLE} | CatBoost available: {CATBOOST_AVAILABLE} | PyTorch available: {TORCH_AVAILABLE}")
    if TORCH_AVAILABLE:
        print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    print(f"Full factorial search: {cfg.full_search}")
    print(f"Parallel n_jobs: {cfg.n_jobs} | backend: {cfg.parallel_backend}")
    print(f"Deep learning enabled: {cfg.enable_deep_learning}")
    print(f"Strict leakage guard: {cfg.strict_leakage_guard} | shared outer splits: {cfg.shared_outer_splits}")
    print(f"Outer split method: {cfg.outer_split_method} | stacking refit within folds: {cfg.strict_stacking_nested_preprocess}")
    print(f"Feature selection: {cfg.feature_selection_mode} | fixed variable count: {cfg.fixed_feature_count}")
    print(f"Outlier-screening scope: {cfg.outlier_filter_scope} (no samples removed before outer splitting)")
    print("==============================\n")


def run_baijiu_model(
    spectral_file: Optional[str] = None,
    phys_file: Optional[str] = None,
    cfg: Optional[Config] = None,
):
    cfg = cfg or Config()
    repo_root = Path(__file__).resolve().parents[1]
    if spectral_file is None:
        spectral_file = str(repo_root / "data" / "raman_sample_mean_spectra_80_samples.xlsx")
    if phys_file is None:
        phys_file = str(repo_root / "data" / "gcms_reference_concentrations_8_esters_80_samples.xlsx")
    if cfg.strict_leakage_guard and not cfg.shared_outer_splits:
        raise ValueError("shared_outer_splits must be True when strict_leakage_guard=True.")
    set_global_seed(cfg.random_state)
    print_environment(cfg)
    wavelengths, sample_names, target_names, X, Y = read_input_tables(spectral_file, phys_file)
    Path(cfg.output_model_dir).mkdir(parents=True, exist_ok=True)
    algs = algorithm_list(cfg)
    validate_factorial_design(cfg, algs)


    provenance = build_provenance_context(
        spectral_file, phys_file, wavelengths, sample_names, target_names, X, Y, cfg
    )
    print("========== Provenance lock ==========")
    print(f"Spectral file SHA256: {provenance['inputFiles']['spectralFileSHA256']}")
    print(f"Reference file SHA256: {provenance['inputFiles']['referenceFileSHA256']}")
    print(f"Paired dataset SHA256: {provenance['alignedData']['pairedDatasetSHA256']}")
    print(f"Script SHA256: {provenance['script']['sha256']}")
    print(f"Torch device mode: {provenance['environment']['torchDeviceMode']}")
    print("=====================================\n")


    all_records, all_best_by_alg, best_summary, completed_targets = load_checkpoint(
        cfg, algs, provenance
    )

    exclusion_manifest = load_predeclared_exclusions(cfg.predeclared_exclusion_file)
    exclusion_audit_rows: List[Dict[str, Any]] = []

    total_start = time.time()
    for t, target in enumerate(target_names):
        target_str = _target_key(target)

        y = Y[:, t]
        keep, population_meta, audit_rows = resolve_target_population(
            target_str, y, sample_names, exclusion_manifest, cfg
        )
        exclusion_audit_rows.extend(audit_rows)


        if cfg.enable_resume and target_str in completed_targets:
            print("=" * 80)
            print(f"Skipping completed analyte: {target_str}")
            print("=" * 80)
            continue

        X0, y0, sn0 = X[keep], y[keep], sample_names[keep]
        records, best_alg, best = run_target(
            target_str, X0, y0, sn0, wavelengths, cfg, population_meta
        )

        all_records.extend(records)
        all_best_by_alg.extend(best_alg)
        if best is not None:
            best_summary.append(best)
            completed_targets.add(target_str)


        if cfg.save_checkpoint_every_target:
            save_checkpoint(cfg, all_records, all_best_by_alg, best_summary, completed_targets, algs, provenance)


        if best_summary:
            original_excel = cfg.output_excel
            tmp_name = Path(original_excel)
            partial_excel = str(tmp_name.with_name(tmp_name.stem + "_partial.xlsx"))
            cfg.output_excel = partial_excel
            try:
                export_excel(best_summary, all_records, all_best_by_alg, cfg, algs, provenance, exclusion_audit_rows)
                print(f"Intermediate results saved: {partial_excel}")
            finally:
                cfg.output_excel = original_excel

    if not best_summary:
        raise RuntimeError("No analyte was modeled successfully; check the input data format.")

    best_df, alg_df, all_df, pred_df = export_excel(
        best_summary, all_records, all_best_by_alg, cfg, algs, provenance, exclusion_audit_rows
    )
    remove_checkpoint_if_finished(cfg)

    print("\n========== Post-selection internal holdout upper-envelope summary for all analytes ==========")
    print(best_df.to_string(index=False))
    print(f"\nExcel results written: {cfg.output_excel}")
    print(f"Checkpoint file: {cfg.checkpoint_file if cfg.enable_resume else 'disabled'}")
    print(f"Total elapsed time: {(time.time() - total_start) / 60:.2f} min")
    return {
        "best_summary": best_df,
        "best_by_algorithm": alg_df,
        "all_run_metrics": all_df,
        "prediction_details": pred_df,
        "records": all_records,
        "best_records": best_summary,
    }


if __name__ == "__main__":
    config = Config(

        full_search=True,

        search_level="balanced",

        n_repeated_split=10,

        n_jobs=6,
        parallel_backend="threading",
        show_realtime_log=True,
        show_detail_each_target=True,
        show_top_n=50,

        enable_deep_learning=True,
        output_excel=str(Path(__file__).resolve().parents[1] / "results" / "raman_baijiu_full_search_results.xlsx"),
        output_model_dir=str(Path(__file__).resolve().parents[1] / "results" / "baijiu_bruteforce_models"),

        enable_resume=True,
        checkpoint_file=str(Path(__file__).resolve().parents[1] / "results" / "raman_baijiu_checkpoint.pkl"),
        save_checkpoint_every_target=True,
        outlier_filter_scope="disabled",
        feature_selection_mode="fixed_count_nested",
        fixed_feature_count=158,
        outer_split_method="response_stratified_random",
        strict_stacking_nested_preprocess=True,
        strict_leakage_guard=True,
        shared_outer_splits=True,
    )
    run_baijiu_model(
        str(Path(__file__).resolve().parents[1] / "data" / "raman_sample_mean_spectra_80_samples.xlsx"),
        str(Path(__file__).resolve().parents[1] / "data" / "gcms_reference_concentrations_8_esters_80_samples.xlsx"),
        config,
    )
