# -*- coding: utf-8 -*-
"""
MCCV异常样本筛选_独立版.py
============================================================
从原始脚本
baijiu_bruteforce_full_search_resume_np_alias_full_fixed_multithread_6(8)(6).py
中单独提取“第一步 MCCV 异常样本筛选”逻辑。

本文件保持原脚本中的核心筛选规则不变：
1. 对每个理化指标分别进行 MCCV；
2. 每次随机取约 78% 样本作为临时校正集；
3. 使用 PLSRegression，最多 12 个潜变量，scale=True；
4. 只在临时预测集上记录绝对预测误差；
5. 对每个样本计算其多次进入预测集时的平均绝对预测误差；
6. 阈值 = 所有样本平均绝对预测误差的中位数 + 3 × MAD；
7. mean_error > threshold 的样本作为候选异常样本；
8. 只有当最终保留样本数 >= 8 且候选异常样本数不超过
   max(10, round(max_outlier_ratio * n)) 时，才正式执行剔除；
   否则保持全部样本。

默认参数与原脚本一致：
    random_state = 2026
    mccv_n_iter = 100
    temporary_calibration_ratio = 0.78
    max_pls_components = 12
    max_outlier_ratio = 0.18

输入数据格式与原脚本一致：
- 光谱文件：
    第1列 = Raman shift / wavelength
    第2列起 = 各样本光谱，列名为样本编号
- 理化指标文件：
    第1列 = 指标名称
    第2列起 = 各样本参考值，列名为样本编号

默认输入：
    MoNiJiu.xlsx
    HuaXueZhiBiao.xlsx

默认输出：
    MCCV异常样本筛选结果.xlsx
    MCCV样本纳入排除清单.csv

依赖：
    pip install numpy pandas scikit-learn openpyxl

运行：
    python MCCV异常样本筛选_独立版.py

或：
    python MCCV异常样本筛选_独立版.py ^
        --spectral "真酒拉曼光谱.xlsx" ^
        --reference "8种酯类理化指标表.xlsx" ^
        --output "MCCV异常样本筛选结果.xlsx"

注意：
- 这是“从原脚本抽离”的版本，不是对 MATLAB mcs.m 的重新实现。
- 该筛选使用完整分析物响应信息后再确定固定样本域，因此若随后再做
  外层校正/预测划分，应在论文中如实说明其属于 response-informed,
  pre-split sample selection。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

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
from sklearn.cross_decomposition import PLSRegression


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
    """沿用原脚本的数据布局和样本对齐逻辑。"""
    print(f"正在读取光谱数据：{spectral_file}")
    print(f"正在读取理化指标数据：{phys_file}")

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

    target_names = phys.iloc[:, 0].astype(str).to_numpy()
    target_names = np.array(
        [
            f"指标{i + 1}"
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

    spec_index = {name: i for i, name in enumerate(spec_sample_names)}
    phys_index = {name: i for i, name in enumerate(phys_sample_names)}

    common = [name for name in spec_sample_names if name in phys_index]
    if len(common) < 8:
        raise ValueError("光谱表与理化表可对齐样本少于 8 个，无法进行 MCCV。")

    X = X_all[[spec_index[n] for n in common], :]
    Y = Y_all[[phys_index[n] for n in common], :]
    sample_names = np.asarray(common, dtype=str)

    print(f"成功对齐样本数：{X.shape[0]}")
    print(f"光谱变量数：{X.shape[1]}")
    print(f"理化指标数：{Y.shape[1]}")
    print()

    return wavelengths, sample_names, target_names, X, Y


def mccv_outlier_diagnostics(
    X: np.ndarray,
    y: np.ndarray,
    cfg: Config,
) -> Dict[str, object]:
    """
    原脚本 mccv_outlier_keep() 的等价独立实现，并额外返回诊断信息。
    核心筛选判定不变。
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()

    n = X.shape[0]
    keep = np.ones(n, dtype=bool)

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

    rng = np.random.default_rng(cfg.random_state)
    pred_err = np.full((cfg.mccv_n_iter, n), np.nan, dtype=float)
    successful_iterations = 0

    for k in range(cfg.mccv_n_iter):
        idx = rng.permutation(n)

        n_train = max(
            8,
            int(round(cfg.temporary_calibration_ratio * n)),
        )

        tr = idx[:n_train]
        te = idx[n_train:]

        if len(te) < 3:
            continue

        try:
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

            pred = model.predict(X[te]).ravel()
            pred_err[k, te] = np.abs(y[te] - pred)
            successful_iterations += 1

        except Exception:
            continue

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

    med = float(np.nanmedian(mean_err))
    mad = float(np.nanmedian(np.abs(mean_err - med)))
    threshold = med + 3.0 * mad

    candidate_outlier = mean_err > threshold

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


def mccv_outlier_keep(
    X: np.ndarray,
    y: np.ndarray,
    cfg: Config,
) -> np.ndarray:
    """与原脚本同名、同用途：仅返回最终 keep 布尔数组。"""
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
    """对理化指标表中的每个指标独立运行 MCCV 异常样本筛选。"""
    _, sample_names, target_names, X, Y = read_input_tables(
        spectral_file,
        reference_file,
    )

    summary_rows: List[Dict[str, object]] = []
    manifest_rows: List[Dict[str, object]] = []
    diagnostic_rows: List[Dict[str, object]] = []

    for t, target in enumerate(target_names):
        y_all = np.asarray(Y[:, t], dtype=float)
        valid = np.isfinite(y_all)

        X0 = X[valid]
        y0 = y_all[valid]
        sn0 = sample_names[valid]

        print("=" * 80)
        print(f"指标：{target}")
        print(f"有效样本数：{len(y0)}")

        if len(y0) < 8 or np.std(y0) < 1e-12:
            print("跳过：样本数不足或参考值无明显变化。")
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

        print(f"MCCV成功子模型数：{diag['successful_iterations']}/{cfg.mccv_n_iter}")
        print(f"median(mean abs error)：{diag['median_error']:.8g}")
        print(f"MAD：{diag['mad']:.8g}")
        print(f"threshold = median + 3*MAD：{diag['threshold']:.8g}")
        print(f"候选异常样本数：{n_candidate}")
        print(f"允许的最大异常样本数：{diag['max_outliers_allowed']}")
        print(f"是否正式执行剔除：{diag['filter_applied']}")
        print(f"最终保留样本数：{n_retained}")
        print(f"最终排除样本数：{n_excluded}")
        if excluded_names:
            print("最终排除样本：", ", ".join(excluded_names))
        print()

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

    summary_df = pd.DataFrame(summary_rows)
    manifest_df = pd.DataFrame(manifest_rows)
    diagnostics_df = pd.DataFrame(diagnostic_rows)

    output_excel_path = Path(output_excel)
    output_excel_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_excel_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        manifest_df.to_excel(writer, sheet_name="Manifest", index=False)
        diagnostics_df.to_excel(writer, sheet_name="Diagnostics", index=False)

    manifest_path = Path(output_manifest_csv)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(
        manifest_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("=" * 80)
    print("MCCV异常样本筛选完成。")
    print(f"Excel结果：{output_excel_path.resolve()}")
    print(f"样本纳入/排除清单：{manifest_path.resolve()}")

    if not summary_df.empty:
        print("\n汇总：")
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="独立运行原始建模脚本第一步中的 MCCV 异常样本筛选。"
    )
    parser.add_argument(
        "--spectral",
        default="MoNiJiu.xlsx",
        help="光谱文件路径（默认：MoNiJiu.xlsx）",
    )
    parser.add_argument(
        "--reference",
        default="HuaXueZhiBiao.xlsx",
        help="理化指标文件路径（默认：HuaXueZhiBiao.xlsx）",
    )
    parser.add_argument(
        "--output",
        default="MCCV异常样本筛选结果.xlsx",
        help="Excel输出文件",
    )
    parser.add_argument(
        "--manifest",
        default="MCCV样本纳入排除清单.csv",
        help="CSV样本纳入/排除清单",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=2026,
        help="随机种子（默认：2026）",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=100,
        help="MCCV迭代次数（默认：100）",
    )
    parser.add_argument(
        "--max-outlier-ratio",
        type=float,
        default=0.18,
        help="原脚本异常比例保护参数（默认：0.18）",
    )
    return parser.parse_args()


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
