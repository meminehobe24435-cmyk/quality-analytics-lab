"""数据质量检查。

这是「数据分析工程师（质量方向）」的核心工作内容，所以做得比较细：
  1. 缺失（按列 / 按行两个维度）
  2. 越界（物理或工艺上下限）
  3. 完全重复行 + 主键重复（含一对多扇出）
  4. 常量列 / 近常量列
  5. 数值列分布漂移（对基线批次的 KS 检验 + PSI 群体稳定性指标）
  6. 类别不平衡比例
  7. 类型与解析问题（数值列里混入非数值）

每项检查都返回**结构化字典**，既能直接 dump 成 JSON，也能转成控制台表格。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from . import config as cfg


# --------------------------------------------------------------------------
# 1. 缺失
# --------------------------------------------------------------------------
def missing_by_column(df: pd.DataFrame) -> pd.DataFrame:
    """按列统计缺失：缺失数、缺失率、以及该列是否**全缺失**。"""
    n = len(df)
    rows = []
    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        rows.append(
            {
                "column": col,
                "dtype": str(df[col].dtype),
                "n_missing": n_missing,
                "pct_missing": round(n_missing / n * 100, 4) if n else 0.0,
                "all_missing": bool(n > 0 and n_missing == n),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["n_missing", "column"], ascending=[False, True]).reset_index(
            drop=True
        )
    return out


def missing_by_row(df: pd.DataFrame) -> dict[str, object]:
    """按行统计缺失：有多少行含缺失、每行缺失个数的分布。"""
    if df.empty:
        return {
            "n_rows": 0,
            "rows_with_any_missing": 0,
            "pct_rows_with_any_missing": 0.0,
            "max_missing_in_a_row": 0,
            "distribution": {},
        }
    miss_per_row = df.isna().sum(axis=1)
    dist = miss_per_row.value_counts().sort_index()
    return {
        "n_rows": int(len(df)),
        "rows_with_any_missing": int((miss_per_row > 0).sum()),
        "pct_rows_with_any_missing": round(float((miss_per_row > 0).mean() * 100), 4),
        "max_missing_in_a_row": int(miss_per_row.max()),
        "distribution": {int(k): int(v) for k, v in dist.items()},
    }


# --------------------------------------------------------------------------
# 2. 越界
# --------------------------------------------------------------------------
def out_of_range_report(
    df: pd.DataFrame, spec_limits: dict[str, tuple[float, float]] | None = None
) -> pd.DataFrame:
    """按列报告越界值：低于下限、高于上限、以及最极端越界值的示例。

    越界只针对存在于 df 且有上下限的数值列；非数值内容会被 to_numeric 转成 NaN
    （这部分由 `dtype_issues` 单独报告，避免与越界混淆）。
    """
    limits = spec_limits if spec_limits is not None else cfg.SPEC_LIMITS
    rows = []
    for col, (low, high) in limits.items():
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        below = values.notna() & (values < low)
        above = values.notna() & (values > high)
        n_bad = int(below.sum() + above.sum())
        extreme = None
        if n_bad:
            bad_values = values.loc[below | above]
            extreme = float(bad_values.iloc[0]) if len(bad_values) else None
        rows.append(
            {
                "column": col,
                "lower": low,
                "upper": high,
                "n_below": int(below.sum()),
                "n_above": int(above.sum()),
                "n_out_of_range": n_bad,
                "pct_out_of_range": round(n_bad / len(df) * 100, 4) if len(df) else 0.0,
                "first_example": extreme,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["n_out_of_range", "column"], ascending=[False, True]
        ).reset_index(drop=True)
    return out


# --------------------------------------------------------------------------
# 3. 重复
# --------------------------------------------------------------------------
def duplicate_report(df: pd.DataFrame, key_columns: tuple[str, ...] = cfg.KEY_COLUMNS) -> dict:
    """完全重复行 + 主键重复统计。

    完全重复行 = 所有列取值都相同的行（检验/录入环节重复上报的典型形态）。
    主键重复 = 本应唯一的键出现多次（如 unit_id 重复 -> 一对多扇出的信号）。
    """
    n = len(df)
    dup_full = int(df.duplicated(keep="first").sum())

    key_cols = [c for c in key_columns if c in df.columns]
    key_stats: dict[str, dict[str, int]] = {}
    if key_cols and n:
        for col in key_cols:
            n_unique = int(df[col].nunique(dropna=True))
            key_stats[col] = {
                "n_unique": n_unique,
                "n_rows": n,
                "n_duplicate_rows_beyond_first": int(n - n_unique),
                "is_unique": bool(n_unique == n),
            }
    return {
        "n_rows": n,
        "n_full_duplicate_rows": dup_full,
        "pct_full_duplicate_rows": round(dup_full / n * 100, 4) if n else 0.0,
        "key_uniqueness": key_stats,
    }


def one_to_many_fanout(
    child: pd.DataFrame, parent_key: str = "unit_id"
) -> dict[str, object]:
    """检查子表（性能测试）相对父表（检验）的一对多扇出情况。

    直接 merge 而不先聚合，会让父表行数被放大，进而把不合格率算错 —— 这是
    多源合并里最容易被忽略的坑，所以单独量化。
    """
    if parent_key not in child.columns:
        return {"parent_key": parent_key, "n_child_rows": int(len(child)), "present": False}
    counts = child.groupby(parent_key).size()
    dist = counts.value_counts().sort_index()
    return {
        "parent_key": parent_key,
        "n_child_rows": int(len(child)),
        "n_distinct_units": int(counts.size),
        "max_tests_per_unit": int(counts.max()) if len(counts) else 0,
        "mean_tests_per_unit": round(float(counts.mean()), 4) if len(counts) else 0.0,
        "rows_if_naively_merged": int(len(child)),
        "distribution": {int(k): int(v) for k, v in dist.items()},
        "fanout_detected": bool(len(counts) and counts.max() > 1),
    }


# --------------------------------------------------------------------------
# 4. 常量 / 近常量列
# --------------------------------------------------------------------------
def constant_column_report(
    df: pd.DataFrame, near_constant_share: float = cfg.NEAR_CONSTANT_SHARE
) -> dict[str, object]:
    """常量列（只有 1 个唯一值）与近常量列（某取值占比 >= 阈值）。

    这两类列对建模没有信息量，且常常是采集系统写死的默认值，属于典型数据质量问题。

    注意：**全缺失列**（有效值 0 个）也算常量列（`value=None`）——
    它同样不含任何信息，而且它同时会出现在缺失检查里（all_missing=True），
    两个检查从不同角度报告同一件事，是刻意保留的冗余。
    """
    constant: list[dict[str, object]] = []
    near: list[dict[str, object]] = []
    for col in df.columns:
        series = df[col].dropna()
        n_unique = int(series.nunique())
        if n_unique <= 1:
            constant.append(
                {
                    "column": col,
                    "n_unique": n_unique,
                    "value": None if series.empty else str(series.iloc[0]),
                    "n_valid": int(len(series)),
                    "all_missing": bool(len(series) == 0 and len(df) > 0),
                }
            )
            continue
        top_share = float(series.value_counts(normalize=True).iloc[0])
        if top_share >= near_constant_share:
            near.append(
                {
                    "column": col,
                    "n_unique": n_unique,
                    "top_value": str(series.value_counts().index[0]),
                    "top_share": round(top_share, 6),
                }
            )
    return {
        "constant_columns": constant,
        "near_constant_columns": sorted(near, key=lambda d: -d["top_share"]),
    }


# --------------------------------------------------------------------------
# 5. 分布漂移
# --------------------------------------------------------------------------
def population_stability_index(
    baseline, current, bins: int = cfg.DRIFT_BINS
) -> float | None:
    """PSI（群体稳定性指标）。

    以基线分位数为分箱边界，比较两组的分布占比：
        PSI = Σ (cur_share - base_share) * ln(cur_share / base_share)
    经验判读：< 0.1 稳定，0.1~0.25 需关注，> 0.25 显著漂移。
    基线无方差（分箱边界退化）或样本为空时返回 None —— 这种情况无法定义漂移。
    """
    b = pd.to_numeric(pd.Series(baseline), errors="coerce").dropna().to_numpy(dtype=float)
    c = pd.to_numeric(pd.Series(current), errors="coerce").dropna().to_numpy(dtype=float)
    if len(b) < 2 or len(c) < 1:
        return None
    edges = np.unique(np.quantile(b, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return None
    edges[0], edges[-1] = -np.inf, np.inf
    base_counts, _ = np.histogram(b, bins=edges)
    cur_counts, _ = np.histogram(c, bins=edges)
    eps = 1e-6
    base_share = np.clip(base_counts / max(len(b), 1), eps, None)
    cur_share = np.clip(cur_counts / max(len(c), 1), eps, None)
    return float(np.sum((cur_share - base_share) * np.log(cur_share / base_share)))


def drift_report(
    df: pd.DataFrame,
    baseline_mask: pd.Series,
    numeric_cols: tuple[str, ...] | None = None,
    alpha: float = cfg.DRIFT_KS_ALPHA,
    psi_threshold: float = cfg.DRIFT_PSI_THRESHOLD,
    ks_stat_threshold: float = cfg.DRIFT_KS_STAT_THRESHOLD,
) -> pd.DataFrame:
    """对基线期 vs 后续期做 KS 检验 + PSI，用**三重门槛**判断漂移。

    物理动机：原料批次之间的差异（含水率、粒径）会真实地造成分布漂移，
    这类漂移必须在建模前被发现，否则模型在新批次上会失效。

    ⚠️ 为什么不能只看 KS 的 p 值（开发中实测踩坑，见 config 里的说明）：
    n≈5000 时 KS 对任意微小差异都显著，实测 11 个列全被判成漂移。
    因此判据是：KS p < alpha **且** KS 统计量 >= 0.20 **且** PSI >= 0.25。
    `drift_severity` 另外给出分档描述（稳定 / 需关注 / 显著漂移）供人工判读。
    """
    cols = numeric_cols if numeric_cols is not None else cfg.ALL_NUMERIC
    baseline_mask = pd.Series(baseline_mask).fillna(False).to_numpy(dtype=bool)
    rows = []
    for col in cols:
        if col not in df.columns:
            continue
        base = pd.to_numeric(df.loc[baseline_mask, col], errors="coerce").dropna()
        rest = pd.to_numeric(df.loc[~baseline_mask, col], errors="coerce").dropna()
        row: dict[str, object] = {
            "column": col,
            "n_baseline": int(len(base)),
            "n_current": int(len(rest)),
            "baseline_mean": round(float(base.mean()), 4) if len(base) else None,
            "current_mean": round(float(rest.mean()), 4) if len(rest) else None,
            "mean_shift_pct": None,
            "ks_statistic": None,
            "ks_p_value": None,
            "ks_significant": False,
            "ks_stat_over_threshold": False,
            "psi": None,
            "psi_over_threshold": False,
            "drift_severity": "样本不足",
            "drifted": False,
            "note": "",
        }
        if len(base) >= 2 and len(rest) >= 2:
            ks = sps.ks_2samp(base, rest)
            row["ks_statistic"] = round(float(ks.statistic), 6)
            row["ks_p_value"] = float(ks.pvalue)
            row["ks_significant"] = bool(ks.pvalue < alpha)
            row["ks_stat_over_threshold"] = bool(float(ks.statistic) >= ks_stat_threshold)
            psi = population_stability_index(base, rest)
            row["psi"] = round(psi, 6) if psi is not None else None
            row["psi_over_threshold"] = bool(psi is not None and psi >= psi_threshold)
            if row["baseline_mean"] not in (None, 0):
                row["mean_shift_pct"] = round(
                    (row["current_mean"] - row["baseline_mean"])
                    / abs(row["baseline_mean"])
                    * 100,
                    4,
                )
            if psi is None:
                row["drift_severity"] = "无法判定（基线无方差）"
                row["note"] = "基线取值几乎全同，PSI 分箱退化"
            elif psi >= 0.25 and row["ks_stat_over_threshold"]:
                row["drift_severity"] = "显著漂移"
            elif psi >= 0.10:
                row["drift_severity"] = "需关注"
            else:
                row["drift_severity"] = "稳定"
            row["drifted"] = bool(
                row["ks_significant"] and row["ks_stat_over_threshold"] and row["psi_over_threshold"]
            )
            if row["ks_significant"] and not row["drifted"]:
                reasons = []
                if not row["ks_stat_over_threshold"]:
                    reasons.append(f"KS 统计量 {row['ks_statistic']} < {ks_stat_threshold}")
                if not row["psi_over_threshold"]:
                    reasons.append(f"PSI {row['psi']} < {psi_threshold}")
                row["note"] = "KS 显著但 " + "、".join(reasons) + "，判为统计显著而工程上不重要"
        else:
            row["note"] = "基线或对照样本不足（< 2），漂移无法判定"
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["drifted", "psi"], ascending=[False, False], na_position="last"
        ).reset_index(drop=True)
    return out




# --------------------------------------------------------------------------
# 6. 类别不平衡
# --------------------------------------------------------------------------
def class_balance(df: pd.DataFrame, label: str = cfg.LABEL_COL) -> dict[str, object]:
    """标签的不平衡比例。少数类占比越低，模型评估越不能只看 accuracy。"""
    if label not in df.columns:
        return {"label": label, "present": False}
    series = pd.to_numeric(df[label], errors="coerce").dropna()
    n = int(len(series))
    if n == 0:
        return {"label": label, "present": True, "n": 0, "note": "标签全为缺失"}
    n_pos = int((series == 1).sum())
    n_neg = int((series == 0).sum())
    pos_rate = n_pos / n
    ratio = (n_neg / n_pos) if n_pos else None
    if n_pos == 0 or n_neg == 0:
        severity = "退化（只有单一类别）"
    elif min(pos_rate, 1 - pos_rate) < 0.05:
        severity = "严重不平衡"
    elif min(pos_rate, 1 - pos_rate) < 0.20:
        severity = "中度不平衡"
    else:
        severity = "基本均衡"
    return {
        "label": label,
        "present": True,
        "n": n,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "positive_rate": round(pos_rate, 6),
        "imbalance_ratio_neg_over_pos": round(ratio, 4) if ratio is not None else None,
        "minority_share": round(min(pos_rate, 1 - pos_rate), 6),
        "severity": severity,
    }


# --------------------------------------------------------------------------
# 7. 类型/解析
# --------------------------------------------------------------------------
def dtype_issues(df: pd.DataFrame, numeric_cols: tuple[str, ...] | None = None) -> list[dict]:
    """数值列里混入非数值内容（录入错误："3.2MPa"、"N/A"、空字符串）。"""
    cols = numeric_cols if numeric_cols is not None else cfg.ALL_NUMERIC
    issues = []
    for col in cols:
        if col not in df.columns:
            continue
        raw = df[col]
        coerced = pd.to_numeric(raw, errors="coerce")
        unparsable = raw.notna() & coerced.isna()
        if unparsable.any():
            examples = raw.loc[unparsable].astype(str).unique()[:3].tolist()
            issues.append(
                {
                    "column": col,
                    "n_unparsable": int(unparsable.sum()),
                    "examples": examples,
                    "declared_dtype": str(raw.dtype),
                }
            )
    return issues


def baseline_mask_by_time(
    df: pd.DataFrame,
    run_col: str = "run_id",
    time_col: str = "start_time",
    fraction: float = cfg.DRIFT_BASELINE_FRACTION,
) -> tuple[pd.Series, dict[str, object]]:
    """按时间挑出漂移基线：最早 `fraction` 比例的炉次视为「历史稳定期」。

    返回 (布尔掩码, 说明)。缺少时间列时退化为「按批次取第一个批次」，
    并在说明里标注，避免悄悄用一个不可靠的基线。
    """
    if time_col in df.columns and run_col in df.columns:
        runs = (
            df[[run_col, time_col]]
            .drop_duplicates(subset=[run_col])
            .sort_values(time_col, kind="stable")
        )
        n_base = max(1, int(round(len(runs) * fraction)))
        base_runs = set(runs[run_col].head(n_base))
        mask = df[run_col].isin(base_runs)
        info = {
            "method": "按 start_time 取最早的炉次",
            "n_baseline_runs": int(n_base),
            "n_total_runs": int(len(runs)),
            "fraction": fraction,
        }
        return mask, info
    if "batch_id" in df.columns:
        first = sorted(df["batch_id"].dropna().unique())[0]
        info = {
            "method": f"缺少时间列，退化为按批次取基线（{first}）",
            "n_baseline_runs": int(df.loc[df["batch_id"] == first, run_col].nunique())
            if run_col in df.columns
            else None,
            "n_total_runs": int(df[run_col].nunique()) if run_col in df.columns else None,
            "fraction": None,
        }
        return df["batch_id"] == first, info
    return pd.Series(False, index=df.index), {"method": "无法确定基线", "fraction": None}


# --------------------------------------------------------------------------
# 汇总
# --------------------------------------------------------------------------
def run_quality_checks(
    wide: pd.DataFrame,
    label: str = cfg.LABEL_COL,
    baseline_batch: str | None = None,
    spec_limits: dict[str, tuple[float, float]] | None = None,
    perf_tests: pd.DataFrame | None = None,
) -> dict[str, object]:
    """跑全部数据质量检查，返回可直接 JSON 化的字典。

    漂移基线默认取「最早的 25% 炉次」；显式传入 `baseline_batch` 时按批次取基线。
    """
    if baseline_batch:
        baseline_mask = (
            wide["batch_id"].astype(str) == baseline_batch
            if "batch_id" in wide.columns
            else pd.Series(False, index=wide.index)
        )
        baseline_info: dict[str, object] = {
            "method": f"按批次取基线（{baseline_batch}）",
            "n_baseline_runs": int(wide.loc[baseline_mask, "run_id"].nunique())
            if "run_id" in wide.columns
            else None,
        }
    else:
        baseline_mask, baseline_info = baseline_mask_by_time(wide)

    report: dict[str, object] = {
        "n_rows": int(len(wide)),
        "n_columns": int(wide.shape[1]),
        "baseline_info": baseline_info,
        "baseline_rows": int(baseline_mask.sum()),
        "missing_by_column": missing_by_column(wide).to_dict(orient="records"),
        "missing_by_row": missing_by_row(wide),
        "out_of_range": out_of_range_report(wide, spec_limits).to_dict(orient="records"),
        "duplicates": duplicate_report(wide),
        "constant_columns": constant_column_report(wide),
        "drift_vs_baseline": drift_report(wide, baseline_mask).to_dict(orient="records"),
        "class_balance": class_balance(wide, label),
        "dtype_issues": dtype_issues(wide),
    }
    if perf_tests is not None:
        report["perf_test_fanout"] = one_to_many_fanout(perf_tests)
    return report



def summarize_quality(report: dict[str, object]) -> dict[str, object]:
    """把质量报告压缩成几个「一行能看懂」的结论，便于报告与断言。"""
    missing_cols = report.get("missing_by_column", [])
    oob = report.get("out_of_range", [])
    const = report.get("constant_columns", {})
    drift = report.get("drift_vs_baseline", [])
    return {
        "n_rows": report.get("n_rows"),
        "columns_with_missing": sum(1 for r in missing_cols if r.get("n_missing")),
        "columns_all_missing": sum(1 for r in missing_cols if r.get("all_missing")),
        "total_out_of_range_values": sum(int(r.get("n_out_of_range", 0)) for r in oob),
        "columns_with_out_of_range": sum(1 for r in oob if r.get("n_out_of_range")),
        "oob_columns": sorted({r["column"] for r in oob if r.get("n_out_of_range")}),
        "oob_detail": {r["column"]: int(r["n_out_of_range"]) for r in oob if r.get("n_out_of_range")},
        "full_duplicate_rows": report.get("duplicates", {}).get("n_full_duplicate_rows"),
        "duplicates": report.get("duplicates", {}),
        "constant_columns": [c["column"] for c in const.get("constant_columns", [])],
        "near_constant_columns": [c["column"] for c in const.get("near_constant_columns", [])],
        "drifted_columns": [r["column"] for r in drift if r.get("drifted")],
        "ks_significant_columns": [r["column"] for r in drift if r.get("ks_significant")],
        "drift_severity": {r["column"]: r.get("drift_severity") for r in drift},
        "class_balance": report.get("class_balance", {}),
        "dtype_issue_columns": [d["column"] for d in report.get("dtype_issues", [])],
    }
