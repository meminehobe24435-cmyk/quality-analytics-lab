"""多源异构数据的落盘、读取与合并。

三路数据 + 一份主数据：
  material_batches   原料批次主数据（批次级，1 行/批次）
  production_runs    生产参数（炉次级，1 行/炉次）
  inspection_results 质量检验结果（单元级，1 行/单元，含标签）
  perf_tests         性能测试（单元级，**1~3 行/单元**，一对多）

合并顺序（保留单元级粒度）：
  inspection_results
    --(unit_id, 1:1)--> perf_tests 先按 unit_id 聚合（均值 + 复测次数）
    --(run_id, n:1)--> production_runs
    --(batch_id, n:1)--> material_batches

关键点：perf_tests 是**一对多**，直接 merge 会让检验行数膨胀（一个单元变多行），
进而把不合格率算错。所以先聚合再连接，并在质量报告里单独报告扇出情况。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg


# --------------------------------------------------------------------------
# 读写
# --------------------------------------------------------------------------
def save_raw_tables(tables: dict[str, pd.DataFrame], data_dir: Path) -> dict[str, Path]:
    """把原始表写成 UTF-8 CSV，返回 {表名: 路径}。"""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name, filename in cfg.RAW_FILES.items():
        path = data_dir / filename
        tables[name].to_csv(path, index=False, encoding="utf-8")
        written[name] = path
    return written


DATETIME_COLUMNS: tuple[str, ...] = (
    "start_time",
    "received_date",
    "inspection_ts",
    "test_ts",
)


def load_raw_tables(data_dir: Path) -> dict[str, pd.DataFrame]:
    """从磁盘读取原始表。

    显式解析时间列（parse_dates）：否则 CSV 里的时间会变成字符串，
    与「刚生成还在内存里」的 datetime64 表示不一致 —— 实测这会让
    数据指纹（sha256）在「重新生成」与「读磁盘」两次运行之间变化，
    从而破坏「同一命令连跑两次结果一致」的可复现性。
    缺文件时抛 FileNotFoundError（调用方决定是否重新生成）。
    """
    data_dir = Path(data_dir)
    tables: dict[str, pd.DataFrame] = {}
    for name, filename in cfg.RAW_FILES.items():
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"缺少原始数据文件: {path}")
        tables[name] = pd.read_csv(
            path,
            encoding="utf-8",
            parse_dates=[c for c in DATETIME_COLUMNS if c in pd.read_csv(path, nrows=0).columns],
        )
    return tables



# --------------------------------------------------------------------------
# 合并
# --------------------------------------------------------------------------
def aggregate_perf_tests(perf: pd.DataFrame) -> pd.DataFrame:
    """把一对多的性能测试聚合到单元级。

    数值列取均值（复测取平均），并额外产出 `perf_test_count`（复测次数），
    复测次数本身可能是有用信息，所以保留而不是丢掉。
    """
    value_cols = [c for c in cfg.PERF_NUMERIC if c in perf.columns]
    agg = perf.groupby("unit_id", as_index=False).agg(
        **{c: (c, "mean") for c in value_cols},
        perf_test_count=("unit_id", "size"),
    )
    return agg


def build_analysis_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """把四张表合并成一张单元级分析宽表。

    返回的宽表以 `unit_id` 为主键，行数应等于检验结果表的行数（含重复行），
    且每个 unit_id 只出现一次（重复行属于「同一 unit_id 重复上报」，是数据质量问题，
    由质量检查模块负责报告，本函数不擅自去重）。
    """
    inspections = tables["inspection_results"].copy()
    runs = tables["production_runs"].copy()
    batches = tables["material_batches"].copy()
    perf = tables["perf_tests"].copy()

    perf_agg = aggregate_perf_tests(perf)

    df = inspections.merge(perf_agg, on="unit_id", how="left", validate="m:1")

    # 检验表里已有 shift（检验班次），生产表里也有 shift（排产班次），合并时区分来源避免歧义。
    # ⚠️ 只重命名 shift：operator_id 与检验表的 inspector_id 不是一回事，
    # 曾经顺手把它一起改名成 run_operator_id，导致它被建模/筛选的列名白名单静默丢掉。
    runs = runs.rename(columns={"shift": "run_shift"})
    df = df.merge(
        runs.drop(columns=["batch_id"]),
        on="run_id",
        how="left",
        validate="m:1",
        suffixes=("", "_run"),
    )
    # 检验班的班次以排产班次为准（二者本应一致，取自同一炉次）
    df = df.drop(columns=["shift"]).rename(columns={"run_shift": "shift"})


    df = df.merge(batches, on="batch_id", how="left", validate="m:1")
    return df


def build_run_level_table(wide: pd.DataFrame, label: str = cfg.LABEL_COL) -> pd.DataFrame:
    """炉次级汇总表：一行一炉次，含不合格率与产出单元数。

    用途：同一炉次内的单元共享潜在风险，单元级检验存在**伪重复**（pseudo-replication），
    炉次级表格是稳健性口径 —— 在炉次级，炉次级因子只有 144 个独立观测。
    """
    factor_numeric = [c for c in cfg.MODEL_NUMERIC if c in wide.columns]
    factor_cat = [c for c in cfg.MODEL_CATEGORICAL if c in wide.columns]

    g = wide.groupby("run_id", as_index=False)
    out = g.agg(
        **{label: (label, "mean")},
        n_units=("unit_id", "size"),
        n_fail=(label, "sum"),
    ).rename(columns={label: "fail_rate"})
    firsts = g[factor_numeric + factor_cat + ["batch_id"]].first()
    # 保存入运行顺序 -> 缩短不同算法下的列拼接歧义
    return out.merge(firsts, on="run_id", how="left", validate="1:1")


def build_batch_level_table(wide: pd.DataFrame, label: str = cfg.LABEL_COL) -> pd.DataFrame:
    """原料批次级汇总表：一行一批次，含不合格率。

    用途：原料因子（含水率、粒径、供应商）只在**批次级**变化，
    同一个批次的多个炉次共享同一个原料值，因此对原料因子而言
    有效样本量是批次数（24），不是炉次数，更不是单元数。
    在炉次级或单元级检验原料因子，同样属于伪重复。
    """
    material_numeric = [c for c in cfg.MATERIAL_NUMERIC if c in wide.columns]
    material_cat = [c for c in ("material_supplier",) if c in wide.columns]

    g = wide.groupby("batch_id", as_index=False)
    out = g.agg(
        **{label: (label, "mean")},
        n_units=("unit_id", "size"),
        n_fail=(label, "sum"),
        n_runs=("run_id", "nunique"),
    ).rename(columns={label: "fail_rate"})
    firsts = g[material_numeric + material_cat].first()
    return out.merge(firsts, on="batch_id", how="left", validate="1:1")



# --------------------------------------------------------------------------
# 清洗
# --------------------------------------------------------------------------
def clean_analysis_table(
    wide: pd.DataFrame, spec_limits: dict[str, tuple[float, float]] | None = None
) -> tuple[pd.DataFrame, dict[str, object]]:
    """按**显式且可复算**的规则清洗宽表，返回 (清洗后宽表, 清洗动作记录)。

    规则（顺序固定，写进报告，便于复算）：
      1. 完全重复行：保留第一次出现，删除其余
      2. 数值列越界（超出物理上下限）：置为 NaN（不直接删行，避免损失该炉次其它信息）
      3. 数值列缺失：用**列中位数**填补（中位数比均值抗离群）
      4. 缺失的类别列：填 "UNKNOWN"
      5. perf 数值列缺失：用列中位数填补，并加 `*_was_missing` 指示列
    每一步的受影响行数都记录在 `actions` 里。
    """
    limits = spec_limits if spec_limits is not None else cfg.SPEC_LIMITS
    df = wide.copy()
    actions: dict[str, object] = {}

    # 1) 完全重复行
    n_before = len(df)
    dup_mask = df.duplicated(keep="first")
    actions["exact_duplicate_rows_dropped"] = int(dup_mask.sum())
    df = df.loc[~dup_mask].reset_index(drop=True)
    actions["rows_before"] = int(n_before)
    actions["rows_after_dedup"] = int(len(df))

    # 2) 越界 -> NaN
    oob_counts: dict[str, int] = {}
    for col, (low, high) in limits.items():
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        oob = values.notna() & ((values < low) | (values > high))
        if oob.any():
            oob_counts[col] = int(oob.sum())
            df.loc[oob, col] = np.nan
    actions["out_of_range_to_nan"] = oob_counts

    # 5) 性能测试缺失指示列（缺失本身可能是信息，如"未测=被视为高风险"）
    # ⚠️ 必须在填补之前记录缺失位置：填完之后再算 isna() 会全是 False（开发中被测试抓到过）
    pre_fill_missing = {}
    for col in df.columns:
        if col.endswith("_was_missing"):
            continue
        flag = f"{col}_was_missing"
        if col in cfg.PERF_NUMERIC and flag not in df.columns:
            pre_fill_missing[flag] = df[col].isna()

    # 3)+4) 缺失填补
    numeric_filled: dict[str, float] = {}
    for col in [*cfg.MODEL_NUMERIC, *cfg.PERF_NUMERIC]:
        if col not in df.columns:
            continue
        if df[col].isna().any():
            median = float(pd.to_numeric(df[col], errors="coerce").median())
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(median)
            numeric_filled[col] = median
    actions["numeric_median_imputed"] = numeric_filled

    cat_filled: list[str] = []
    for col in cfg.MODEL_CATEGORICAL:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].astype("object").where(df[col].notna(), "UNKNOWN")
            cat_filled.append(col)
    actions["categorical_filled_unknown"] = cat_filled

    for flag, mask in pre_fill_missing.items():
        df[flag] = mask.to_numpy(dtype=bool)
    missing_flags = sorted(pre_fill_missing)
    actions["missing_indicator_columns"] = missing_flags

    # 标签必须完整，缺失标签的行无法用于监督学习
    if cfg.LABEL_COL in df.columns:
        bad_label = df[cfg.LABEL_COL].isna()
        actions["rows_dropped_missing_label"] = int(bad_label.sum())
        if bad_label.any():
            df = df.loc[~bad_label].reset_index(drop=True)
    actions["rows_final"] = int(len(df))
    return df, actions


def frame_fingerprint(df: pd.DataFrame) -> str:
    """宽表内容指纹（sha256）。

    用于把「本次分析基于哪份数据」写进指标文件：同一命令连跑两次，
    指纹必须一致；数据被改动则指纹变化。
    """
    import hashlib

    hashed = pd.util.hash_pandas_object(df, index=True).to_numpy()
    return hashlib.sha256(hashed.tobytes()).hexdigest()
