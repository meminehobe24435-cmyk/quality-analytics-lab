"""合成数据生成器的测试。

重点：注入的每一个已知异常都必须真的被注入（数量精确），
生成过程必须可复现，并且「设计的主效应」在数据里真的存在。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qalab import config as cfg
from qalab import synth


def test_generate_all_returns_four_tables(tables):
    assert set(tables) == {
        "material_batches",
        "production_runs",
        "inspection_results",
        "perf_tests",
    }
    assert len(tables["material_batches"]) == cfg.N_BATCHES
    assert len(tables["production_runs"]) == cfg.N_BATCHES * cfg.RUNS_PER_BATCH
    assert len(tables["inspection_results"]) > 0
    assert len(tables["perf_tests"]) > 0


def test_generation_is_reproducible():
    a, _ = synth.generate_all()
    b, _ = synth.generate_all()
    for name in a:
        pd.testing.assert_frame_equal(a[name], b[name])


def test_different_seed_gives_different_data():
    a, _ = synth.generate_all(cfg.SEED)
    b, _ = synth.generate_all(cfg.SEED + 12345)
    assert not a["inspection_results"]["is_fail"].equals(b["inspection_results"]["is_fail"])


def test_label_has_both_classes_and_plausible_rate(tables):
    insp = tables["inspection_results"]
    rate = insp["is_fail"].mean()
    assert set(insp["is_fail"].unique()) == {0, 1}
    # 校准目标约 15%，允许 8%~25%
    assert 0.08 < rate < 0.25, f"不合格率 {rate} 超出校准范围"


def test_unit_ids_are_unique_except_injected_duplicates(tables):
    """unit_id 本应唯一；被注入的重复行会让它不再唯一 —— 这正是重复检查要抓的问题。"""
    insp = tables["inspection_results"]
    n_dup_units = int(insp["unit_id"].duplicated().sum())
    assert n_dup_units == cfg.N_INJECTED_DUPLICATE_ROWS
    assert insp["unit_id"].nunique() == len(insp) - cfg.N_INJECTED_DUPLICATE_ROWS


def test_inspection_rows_exist_for_every_run(tables):
    runs = set(tables["production_runs"]["run_id"])
    for col in ("defect_type", "is_fail", "inspector_id", "inspection_method"):
        assert col in tables["inspection_results"].columns
    # 每张检验记录都要能挂到一个炉次上
    assert "run_id" in tables["inspection_results"].columns
    assert len(runs) == cfg.N_BATCHES * cfg.RUNS_PER_BATCH


def test_perf_tests_are_one_to_many(tables):
    perf = tables["perf_tests"]
    counts = perf.groupby("unit_id").size()
    assert counts.max() >= 2, "应当存在复测（一对多）"
    assert counts.size < len(perf), "性能测试行数应多于被测单元数"
    assert perf["unit_id"].nunique() == counts.size


def test_fail_units_have_non_none_defect_type(tables):
    insp = tables["inspection_results"]
    fails = insp[insp["is_fail"] == 1]
    passes = insp[insp["is_fail"] == 0]
    assert (fails["defect_type"] != "NONE").all()
    assert (passes["defect_type"] == "NONE").all()


def test_injected_out_of_range_furnace_temp(tables, manifest):
    runs = tables["production_runs"]
    injected = runs[runs["furnace_temp_c"] == cfg.INJECTED_OUT_OF_RANGE_VALUE]
    assert len(injected) == cfg.N_INJECTED_OUT_OF_RANGE
    assert set(injected["run_id"]) == set(manifest["out_of_range_furnace_temp"]["runs"])
    # 除了注入的 9999，其余炉温都必须落在规格内
    others = runs[runs["furnace_temp_c"] != cfg.INJECTED_OUT_OF_RANGE_VALUE]
    low, high = cfg.SPEC_LIMITS["furnace_temp_c"]
    assert others["furnace_temp_c"].between(low, high).all()


def test_injected_missing_values(tables, manifest):
    assert (
        int(tables["inspection_results"]["inspector_id"].isna().sum())
        == cfg.N_INJECTED_MISSING_INSPECTOR
    )
    assert int(tables["production_runs"]["machine_id"].isna().sum()) == cfg.N_INJECTED_MISSING_MACHINE
    assert manifest["missing_inspector_id"] == cfg.N_INJECTED_MISSING_INSPECTOR
    assert manifest["missing_machine_id"] == cfg.N_INJECTED_MISSING_MACHINE


def test_injected_duplicate_rows(tables, manifest):
    insp = tables["inspection_results"]
    n_dup = int(insp.duplicated(keep="first").sum())
    assert n_dup == cfg.N_INJECTED_DUPLICATE_ROWS
    assert manifest["duplicate_inspection_rows"] == cfg.N_INJECTED_DUPLICATE_ROWS


def test_injected_constant_and_near_constant_columns(tables, manifest):
    insp = tables["inspection_results"]
    assert insp[cfg.CONSTANT_COLUMN_NAME].nunique() == 1
    assert insp[cfg.CONSTANT_COLUMN_NAME].iloc[0] == cfg.CONSTANT_COLUMN_VALUE
    share = insp[cfg.NEAR_CONSTANT_COLUMN_NAME].mean()
    assert share < 0.01
    assert share >= cfg.NEAR_CONSTANT_TRUE_RATE * 0.5
    assert manifest["constant_column"] == cfg.CONSTANT_COLUMN_NAME


def test_injected_out_of_spec_perf_readings(tables, manifest):
    perf = tables["perf_tests"]
    low, high = cfg.SPEC_LIMITS["tensile_strength_mpa"]
    n_below = int((perf["tensile_strength_mpa"] < low).sum())
    assert n_below == cfg.N_INJECTED_OOS_PERF
    assert manifest["out_of_spec_perf_readings"]["n"] == cfg.N_INJECTED_OOS_PERF


def test_perf_values_mostly_inside_spec_limits(tables):
    """过程能力经过校准：绝大多数性能测试值必须落在规格内（越界是异常而不是常态）。"""
    perf = tables["perf_tests"]
    for col, (low, high) in cfg.SPEC_LIMITS.items():
        if col not in perf.columns:
            continue
        values = perf[col].dropna()
        inside = ((values >= low) & (values <= high)).mean()
        assert inside > 0.97, f"{col} 落在规格内的比例只有 {inside:.3f}"


def test_material_batch_effect_exists(tables):
    """批次效应的物理动机：含水率必须在批次之间有真实差异（否则漂移检查没意义）。"""
    batches = tables["material_batches"]
    assert batches["material_moisture_pct"].std() > 0.15
    # 不同供应商的含水率均值应当不同（设计上 SUP-C 最高）
    means = batches.groupby("material_supplier")["material_moisture_pct"].mean()
    if {"SUP-A", "SUP-C"}.issubset(means.index):
        assert means["SUP-C"] > means["SUP-A"]


def test_designed_primary_effects_exist_in_raw_data(tables):
    """真值检查：设计的三个主因子必须与标签有不弱于 0.15 的秩相关。

    这只验证「数据里确实有信号」，不验证筛选算法（那在 test_significance.py）。
    """
    runs = tables["production_runs"]
    insp = tables["inspection_results"]
    perf = tables["perf_tests"]
    df = insp.merge(runs.drop(columns=["batch_id"]), on="run_id").merge(
        tables["material_batches"], on="batch_id"
    )
    # 排除注入的传感器故障值，否则会污染相关性
    temp = df["furnace_temp_c"].where(df["furnace_temp_c"] < 2000)
    for col, sign in (
        ("furnace_temp_c", 1),
        ("pressure_mpa", 1),
        ("material_moisture_pct", 1),
    ):
        values = temp if col == "furnace_temp_c" else df[col]
        rho = values.corr(df["is_fail"], method="spearman")
        assert rho * sign > 0.15, f"{col} 与标签的秩相关只有 {rho:.3f}（符号应为 {sign}）"
    assert len(perf) > 0
