"""数据质量检查的测试。

每个检查项都要覆盖边界：全缺失列、单一类别、越界值、重复行、常量列、
退化输入（空表、无方差）、以及漂移判据的三重门槛。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qalab import config as cfg
from qalab import quality as q


# --------------------------------------------------------------------------
# 构造小而可控的输入
# --------------------------------------------------------------------------
def _toy() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "run_id": ["R1", "R1", "R2", "R2"],
            "batch_id": ["B1", "B1", "B2", "B2"],
            "start_time": pd.to_datetime(
                ["2025-01-01 06:00", "2025-01-01 06:00", "2025-02-01 06:00", "2025-02-01 06:00"]
            ),
            "temp": [100.0, 101.0, 102.0, 103.0],
            "pressure": [1.0, 2.0, np.nan, 4.0],
            "empty": [np.nan, np.nan, np.nan, np.nan],
            "cat": ["A", "A", "B", "B"],
            "single": ["X", "X", "X", "X"],
            "is_fail": [0, 1, 0, 1],
        }
    )


# --------------------------------------------------------------------------
# 缺失
# --------------------------------------------------------------------------
def test_missing_by_column_counts_and_all_missing_flag():
    out = q.missing_by_column(_toy()).set_index("column")
    assert out.loc["pressure", "n_missing"] == 1
    assert out.loc["empty", "n_missing"] == 4
    assert out.loc["empty", "pct_missing"] == pytest.approx(100.0)
    assert bool(out.loc["empty", "all_missing"]) is True
    assert bool(out.loc["temp", "all_missing"]) is False
    assert out.loc["temp", "n_missing"] == 0


def test_missing_by_column_on_empty_frame():
    out = q.missing_by_column(pd.DataFrame({"a": [], "b": []}))
    assert (out["n_missing"] == 0).all()
    assert not out["all_missing"].any()


def test_missing_by_row_distribution():
    out = q.missing_by_row(_toy())
    assert out["n_rows"] == 4
    assert out["rows_with_any_missing"] == 4  # empty 列全缺失，所以每行都有缺失
    assert out["max_missing_in_a_row"] == 2  # 第 3 行同时缺 pressure 与 empty
    assert out["distribution"] == {1: 3, 2: 1}
    assert out["pct_rows_with_any_missing"] == pytest.approx(100.0)


def test_missing_by_row_on_empty_frame():
    out = q.missing_by_row(pd.DataFrame())
    assert out["n_rows"] == 0
    assert out["rows_with_any_missing"] == 0
    assert out["distribution"] == {}


# --------------------------------------------------------------------------
# 越界
# --------------------------------------------------------------------------
def test_out_of_range_counts_below_and_above():
    df = pd.DataFrame({"furnace_temp_c": [1500.0, 1400.0, 1600.0, 1505.0]})
    out = q.out_of_range_report(df).set_index("column")
    low, high = cfg.SPEC_LIMITS["furnace_temp_c"]
    assert low == 1420.0 and high == 1580.0
    assert out.loc["furnace_temp_c", "n_below"] == 1
    assert out.loc["furnace_temp_c", "n_above"] == 1
    assert out.loc["furnace_temp_c", "n_out_of_range"] == 2
    assert out.loc["furnace_temp_c", "first_example"] == pytest.approx(1400.0)


def test_out_of_range_detects_injected_sensor_fault(wide):
    out = q.out_of_range_report(wide).set_index("column")
    n_above = out.loc["furnace_temp_c", "n_above"]
    assert n_above > 0
    assert out.loc["furnace_temp_c", "first_example"] == cfg.INJECTED_OUT_OF_RANGE_VALUE
    # 越界数量应等于注入炉次产出的单元数
    injected_units = wide[wide["run_id"].isin(
        [f"R{i:03d}" for i in cfg.INJECTED_OUT_OF_RANGE_RUNS]
    )]
    assert n_above == len(injected_units)


def test_out_of_range_ignores_columns_without_limits():
    df = pd.DataFrame({"unknown_col": [1e9, -1e9]})
    out = q.out_of_range_report(df)
    assert out.empty


def test_out_of_range_empty_frame():
    out = q.out_of_range_report(pd.DataFrame({"furnace_temp_c": []}))
    assert out["n_out_of_range"].sum() == 0


# --------------------------------------------------------------------------
# 重复
# --------------------------------------------------------------------------
def test_duplicate_report_counts_full_and_key_duplicates():
    df = _toy()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    rep = q.duplicate_report(df)
    assert rep["n_full_duplicate_rows"] == 1
    assert rep["key_uniqueness"]["run_id"]["n_unique"] == 2
    assert rep["key_uniqueness"]["run_id"]["is_unique"] is False


def test_duplicate_report_on_unique_keys(clean_wide):
    rep = q.duplicate_report(clean_wide)
    assert rep["key_uniqueness"]["unit_id"]["is_unique"] is True
    assert rep["n_full_duplicate_rows"] == 0


def test_one_to_many_fanout_detects_repeats(tables):
    fan = q.one_to_many_fanout(tables["perf_tests"])
    assert fan["fanout_detected"] is True
    assert fan["max_tests_per_unit"] == 3
    assert fan["n_distinct_units"] < fan["n_child_rows"]
    assert fan["distribution"][1] > 0


def test_one_to_many_fanout_without_repeats():
    child = pd.DataFrame({"unit_id": ["U1", "U2", "U3"]})
    fan = q.one_to_many_fanout(child)
    assert fan["fanout_detected"] is False
    assert fan["max_tests_per_unit"] == 1


def test_one_to_many_fanout_missing_key():
    fan = q.one_to_many_fanout(pd.DataFrame({"other": [1, 2]}))
    assert fan["present"] is False


# --------------------------------------------------------------------------
# 常量 / 近常量
# --------------------------------------------------------------------------
def test_constant_and_near_constant_columns():
    rep = q.constant_column_report(_toy())
    const_names = [c["column"] for c in rep["constant_columns"]]
    # 'single' 只有一个取值；'empty' 全缺失（零信息，也算常量列）
    assert "single" in const_names
    assert "empty" in const_names
    entry = next(c for c in rep["constant_columns"] if c["column"] == "single")
    assert entry["value"] == "X"
    assert entry["n_valid"] == 4
    all_missing_entry = next(c for c in rep["constant_columns"] if c["column"] == "empty")
    assert all_missing_entry["value"] is None
    assert all_missing_entry["all_missing"] is True
    # 有 2 个取值的列不应被判为常量
    assert "cat" not in const_names


def test_constant_column_with_single_category_detected(clean_wide):
    rep = q.constant_column_report(clean_wide)
    const_names = [c["column"] for c in rep["constant_columns"]]
    assert cfg.CONSTANT_COLUMN_NAME in const_names
    near = [c["column"] for c in rep["near_constant_columns"]]
    assert cfg.NEAR_CONSTANT_COLUMN_NAME in near
    entry = next(c for c in rep["near_constant_columns"] if c["column"] == cfg.NEAR_CONSTANT_COLUMN_NAME)
    assert entry["top_share"] > cfg.NEAR_CONSTANT_SHARE


def test_near_constant_threshold_is_respected():
    # 90% 同一个值时不应被判为近常量（阈值 0.99）
    df = pd.DataFrame({"x": ["A"] * 90 + ["B"] * 10})
    rep = q.constant_column_report(df)
    assert rep["near_constant_columns"] == []
    assert rep["constant_columns"] == []
    # 99.5% 同一个值则应被判为近常量
    df2 = pd.DataFrame({"x": ["A"] * 199 + ["B"]})
    rep2 = q.constant_column_report(df2)
    assert [c["column"] for c in rep2["near_constant_columns"]] == ["x"]


# --------------------------------------------------------------------------
# 漂移
# --------------------------------------------------------------------------
def test_psi_is_near_zero_for_identical_distribution():
    rng = np.random.default_rng(0)
    a = rng.normal(10, 2, 5000)
    psi = q.population_stability_index(a, a.copy())
    assert psi is not None and psi < 0.01


def test_psi_is_large_for_shifted_distribution():
    rng = np.random.default_rng(1)
    base = rng.normal(10, 2, 5000)
    shifted = rng.normal(13, 2, 5000)
    psi = q.population_stability_index(base, shifted)
    assert psi is not None and psi > 0.5


def test_psi_returns_none_when_baseline_has_no_variance():
    assert q.population_stability_index([5.0] * 100, [1.0, 2.0, 3.0]) is None
    assert q.population_stability_index([], [1.0]) is None
    assert q.population_stability_index([1.0], [1.0]) is None


def test_drift_report_requires_all_three_gates():
    """KS 显著但效应量/PSI 不达标时，不得判为漂移（这是开发中踩过的坑）。"""
    rng = np.random.default_rng(7)
    base = rng.normal(0, 1, 3000)
    # 极微小但统计显著的差异（n 很大 -> KS p 极小，KS 统计量与 PSI 都很小）
    tiny = rng.normal(0.03, 1, 3000)
    df = pd.DataFrame(
        {
            "run_id": ["R1"] * 3000 + ["R2"] * 3000,
            "start_time": [pd.Timestamp("2025-01-01")] * 3000 + [pd.Timestamp("2025-02-01")] * 3000,
            "furnace_temp_c": np.concatenate([base, tiny]),
        }
    )
    mask = df["run_id"] == "R1"
    out = q.drift_report(df, mask)
    row = out.iloc[0]
    assert bool(row["ks_significant"]) is True
    assert bool(row["drifted"]) is False
    assert "KS 显著但" in row["note"]


def test_drift_report_flags_large_shift():
    rng = np.random.default_rng(8)
    base = rng.normal(0, 1, 3000)
    big = rng.normal(2.0, 1, 3000)
    df = pd.DataFrame(
        {
            "run_id": ["R1"] * 3000 + ["R2"] * 3000,
            "start_time": [pd.Timestamp("2025-01-01")] * 3000 + [pd.Timestamp("2025-02-01")] * 3000,
            "furnace_temp_c": np.concatenate([base, big]),
        }
    )
    out = q.drift_report(df, df["run_id"] == "R1")
    row = out.iloc[0]
    assert bool(row["drifted"]) is True
    assert row["drift_severity"] == "显著漂移"
    assert row["mean_shift_pct"] is not None


def test_drift_report_handles_insufficient_samples():
    df = pd.DataFrame({"run_id": ["R1", "R2"], "furnace_temp_c": [1500.0, 1501.0]})
    out = q.drift_report(df, df["run_id"] == "R1")
    assert bool(out.iloc[0]["drifted"]) is False
    assert out.iloc[0]["psi"] is None
    assert "样本不足" in out.iloc[0]["note"]


def test_baseline_mask_by_time_picks_earliest_runs(clean_wide):
    mask, info = q.baseline_mask_by_time(clean_wide, fraction=0.25)
    assert info["method"].startswith("按 start_time")
    assert info["n_baseline_runs"] == int(round(info["n_total_runs"] * 0.25))
    base_runs = set(clean_wide.loc[mask, "run_id"])
    assert len(base_runs) == info["n_baseline_runs"]
    # 基线炉次的开始时间必须早于非基线炉次
    all_runs = clean_wide.groupby("run_id")["start_time"].first()
    assert all_runs[list(base_runs)].max() < all_runs[
        [r for r in all_runs.index if r not in base_runs]
    ].min()


def test_baseline_mask_falls_back_to_batch_when_no_time():
    df = pd.DataFrame({"run_id": ["R1", "R2"], "batch_id": ["B02", "B01"], "x": [1.0, 2.0]})
    mask, info = q.baseline_mask_by_time(df)
    assert "退化为按批次" in info["method"]
    assert mask.tolist() == [False, True]


# --------------------------------------------------------------------------
# 类别不平衡
# --------------------------------------------------------------------------
def test_class_balance_reports_ratio_and_severity():
    df = pd.DataFrame({"is_fail": [0] * 88 + [1] * 12})
    bal = q.class_balance(df)
    assert bal["n_positive"] == 12
    assert bal["n_negative"] == 88
    assert bal["positive_rate"] == pytest.approx(0.12)
    assert bal["imbalance_ratio_neg_over_pos"] == pytest.approx(round(88 / 12, 4))
    assert bal["severity"] == "中度不平衡"


def test_class_balance_severity_levels():
    assert q.class_balance(pd.DataFrame({"is_fail": [0] * 50 + [1] * 50}))["severity"] == "基本均衡"
    assert q.class_balance(pd.DataFrame({"is_fail": [0] * 97 + [1] * 3}))["severity"] == "严重不平衡"
    single = q.class_balance(pd.DataFrame({"is_fail": [0] * 10}))
    assert "退化" in single["severity"]
    assert single["imbalance_ratio_neg_over_pos"] is None


def test_class_balance_missing_label_column():
    assert q.class_balance(pd.DataFrame({"x": [1]}))["present"] is False


def test_class_balance_all_label_missing():
    bal = q.class_balance(pd.DataFrame({"is_fail": [np.nan, np.nan]}))
    assert bal["n"] == 0
    assert "全为缺失" in bal["note"]


# --------------------------------------------------------------------------
# 类型问题
# --------------------------------------------------------------------------
def test_dtype_issues_detects_unparsable_numbers():
    df = pd.DataFrame({"furnace_temp_c": ["1500", "3.2MPa", None, "1480"]})
    issues = q.dtype_issues(df)
    assert len(issues) == 1
    assert issues[0]["column"] == "furnace_temp_c"
    assert issues[0]["n_unparsable"] == 1
    assert "3.2MPa" in issues[0]["examples"]


def test_dtype_issues_clean_numeric_column_has_no_issues():
    assert q.dtype_issues(pd.DataFrame({"furnace_temp_c": [1500.0, 1501.0]})) == []


# --------------------------------------------------------------------------
# 汇总
# --------------------------------------------------------------------------
def test_run_quality_checks_has_all_sections(wide, tables):
    rep = q.run_quality_checks(wide, perf_tests=tables["perf_tests"])
    for key in (
        "missing_by_column",
        "missing_by_row",
        "out_of_range",
        "duplicates",
        "constant_columns",
        "drift_vs_baseline",
        "class_balance",
        "dtype_issues",
        "perf_test_fanout",
        "baseline_info",
    ):
        assert key in rep, f"质量报告缺少 {key}"
    assert rep["n_rows"] == len(wide)


def test_summarize_quality_matches_injected_anomalies(wide, tables):
    rep = q.run_quality_checks(wide, perf_tests=tables["perf_tests"])
    s = q.summarize_quality(rep)
    assert s["full_duplicate_rows"] == cfg.N_INJECTED_DUPLICATE_ROWS
    assert cfg.CONSTANT_COLUMN_NAME in s["constant_columns"]
    assert cfg.NEAR_CONSTANT_COLUMN_NAME in s["near_constant_columns"]
    assert s["total_out_of_range_values"] >= cfg.N_INJECTED_OUT_OF_RANGE
    assert "furnace_temp_c" in s["oob_columns"]
    assert s["columns_all_missing"] == 0
    assert s["columns_with_missing"] > 0
    assert s["dtype_issue_columns"] == []


def test_quality_report_is_json_serializable(wide, tables):
    import json

    rep = q.run_quality_checks(wide, perf_tests=tables["perf_tests"])
    text = json.dumps(rep, ensure_ascii=False, default=str)
    assert len(text) > 100
