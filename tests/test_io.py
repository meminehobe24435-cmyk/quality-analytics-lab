"""多源合并与清洗的测试。

覆盖：一对多聚合、行数守恒、键唯一性、清洗各项动作的数量与语义、
数据指纹的稳定性与敏感性。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qalab import config as cfg
from qalab import io as qio


def test_aggregate_perf_tests_collapses_to_unit_level(tables):
    agg = qio.aggregate_perf_tests(tables["perf_tests"])
    assert agg["unit_id"].is_unique
    assert agg["unit_id"].nunique() == tables["perf_tests"]["unit_id"].nunique()
    assert (agg["perf_test_count"] >= 1).all()
    assert agg["perf_test_count"].sum() == len(tables["perf_tests"])


def test_aggregate_takes_mean_of_repeats():
    perf = pd.DataFrame(
        {
            "unit_id": ["U1", "U1", "U2"],
            "tensile_strength_mpa": [400.0, 500.0, 430.0],
            "hardness_hv": [100.0, 120.0, 110.0],
            "elongation_pct": [8.0, 10.0, 9.0],
            "surface_roughness_ra": [1.0, 1.4, 1.2],
        }
    )
    agg = qio.aggregate_perf_tests(perf).set_index("unit_id")
    assert agg.loc["U1", "tensile_strength_mpa"] == pytest.approx(450.0)
    assert agg.loc["U1", "perf_test_count"] == 2
    assert agg.loc["U2", "tensile_strength_mpa"] == pytest.approx(430.0)


def test_merge_preserves_inspection_row_count(wide, tables):
    """关键不变量：一对多先聚合再连接，宽表行数必须等于检验表行数。

    如果直接把 perf_tests 拿去 merge（不聚合），行数会膨胀，不合格率算错。
    """
    assert len(wide) == len(tables["inspection_results"])
    naive = tables["inspection_results"].merge(
        tables["perf_tests"], on="unit_id", how="left"
    )
    assert len(naive) > len(wide), "朴素 merge 应当膨胀行数（这就是要避免的坑）"


def test_wide_table_has_expected_columns(wide):
    for col in (
        *cfg.PRODUCTION_NUMERIC,
        *cfg.MATERIAL_NUMERIC,
        *cfg.PRODUCTION_CATEGORICAL,
        *cfg.PERF_NUMERIC,
        cfg.LABEL_COL,
        "batch_id",
        "run_id",
        "unit_id",
    ):
        assert col in wide.columns, f"宽表缺少列 {col}"
    # 只应存在一个 shift 列（来自排产），不应出现 shift_run 之类的残渣
    assert "shift" in wide.columns
    assert "run_shift" not in wide.columns
    assert "operator_id" in wide.columns


def test_every_unit_row_maps_to_a_known_run(wide, tables):
    known_runs = set(tables["production_runs"]["run_id"])
    assert set(wide["run_id"]).issubset(known_runs)
    known_batches = set(tables["material_batches"]["batch_id"])
    assert set(wide["batch_id"]).issubset(known_batches)


def test_perf_columns_missing_for_unsampled_units(wide):
    missing_frac = wide["tensile_strength_mpa"].isna().mean()
    # 抽检比例 0.7 -> 约 30% 的单元没有性能测试数据
    assert 0.15 < missing_frac < 0.45


def test_clean_drops_exact_duplicate_rows(wide):
    clean, actions = qio.clean_analysis_table(wide)
    assert actions["exact_duplicate_rows_dropped"] == cfg.N_INJECTED_DUPLICATE_ROWS
    assert len(clean) == len(wide) - cfg.N_INJECTED_DUPLICATE_ROWS
    assert not clean.duplicated().any()


def test_clean_replaces_out_of_range_with_median_then_imputes(wide):
    clean, actions = qio.clean_analysis_table(wide)
    low, high = cfg.SPEC_LIMITS["furnace_temp_c"]
    # 清洗后不应再有越界值
    assert clean["furnace_temp_c"].between(low, high).all()
    assert actions["out_of_range_to_nan"]["furnace_temp_c"] > 0
    # 被填补的列必须记录中位数
    assert "furnace_temp_c" in actions["numeric_median_imputed"]
    median = actions["numeric_median_imputed"]["furnace_temp_c"]
    assert low <= median <= high


def test_clean_keeps_rows_and_only_drops_duplicates(wide):
    clean, actions = qio.clean_analysis_table(wide)
    assert actions["rows_final"] == len(clean)
    assert actions["rows_after_dedup"] == len(wide) - cfg.N_INJECTED_DUPLICATE_ROWS
    assert actions["rows_dropped_missing_label"] == 0


def test_clean_fills_missing_categorical_with_unknown(wide):
    clean, actions = qio.clean_analysis_table(wide)
    assert "machine_id" in actions["categorical_filled_unknown"]
    assert not clean["machine_id"].isna().any()
    assert (clean["machine_id"] == "UNKNOWN").sum() > 0
    for col in cfg.MODEL_CATEGORICAL:
        assert not clean[col].isna().any()


def test_missing_indicator_columns_capture_pre_fill_missing(wide):
    """缺失指示列必须在填补**之前**记录，否则会全为 False（开发中踩过这个坑）。

    指示列的语义是「这条记录的该测量值不可用」，因此它同时覆盖
    「原本就缺失」和「越界被置为缺失」两种情况。
    """
    clean, actions = qio.clean_analysis_table(wide)
    flags = actions["missing_indicator_columns"]
    assert flags, "应当生成缺失指示列"
    for flag in flags:
        assert clean[flag].dtype == bool
    frac_missing = clean["tensile_strength_mpa_was_missing"].mean()
    assert 0.15 < frac_missing < 0.45, f"指示列占比 {frac_missing} 不合理（可能全 False）"

    low, high = cfg.SPEC_LIMITS["tensile_strength_mpa"]
    original = wide["tensile_strength_mpa"]
    out_of_range = original.notna() & ((original < low) | (original > high))
    expected = (original.isna() | out_of_range).sum()
    assert clean["tensile_strength_mpa_was_missing"].sum() == expected
    assert int(out_of_range.sum()) > 0


def test_repeat_averaging_dilutes_single_bad_reading(wide, tables):
    """一对多聚合会把「单条越界读数」稀释掉：6 条越界读数 -> 只剩 3 个越界单元。

    这是真实的工程含义：性能测试复测取平均之后，单次抽检的异常会被平均掉，
    如果质检流程只看均值，就会漏掉「测过一次但不合格」的单元。
    """
    low, high = cfg.SPEC_LIMITS["tensile_strength_mpa"]
    perf = tables["perf_tests"]
    n_oos_readings = int((perf["tensile_strength_mpa"] < low).sum())
    assert n_oos_readings == cfg.N_INJECTED_OOS_PERF

    n_oos_units = int((wide["tensile_strength_mpa"] < low).sum())
    assert 0 < n_oos_units < n_oos_readings
    # 被稀释掉的单元，其复测里确实有一条越界读数
    diluted = wide.loc[wide["tensile_strength_mpa"] >= low, "unit_id"]
    oos_units = set(perf.loc[perf["tensile_strength_mpa"] < low, "unit_id"])
    assert oos_units - set(diluted) != set()
    assert len(oos_units) > n_oos_units



def test_clean_does_not_touch_label(wide):
    clean, _ = qio.clean_analysis_table(wide)
    assert clean[cfg.LABEL_COL].isin([0, 1]).all()
    expected = wide[~wide.duplicated(keep="first")][cfg.LABEL_COL].sum()
    assert clean[cfg.LABEL_COL].sum() == expected


def test_run_level_table_groups_units_correctly(clean_wide, run_level):
    assert len(run_level) == clean_wide["run_id"].nunique()
    assert run_level["run_id"].is_unique
    assert np.isclose(run_level["n_units"].sum(), len(clean_wide))
    assert run_level["fail_rate"].between(0, 1).all()
    row = run_level.iloc[0]
    subset = clean_wide[clean_wide["run_id"] == row["run_id"]]
    assert row["n_units"] == len(subset)
    assert np.isclose(row["fail_rate"], subset[cfg.LABEL_COL].mean())


def test_batch_level_table_groups_runs_correctly(clean_wide, batch_level):
    assert len(batch_level) == clean_wide["batch_id"].nunique()
    assert batch_level["batch_id"].is_unique
    assert np.isclose(batch_level["n_units"].sum(), len(clean_wide))
    total_runs = batch_level["n_runs"].sum()
    assert total_runs >= clean_wide["run_id"].nunique()
    assert batch_level["fail_rate"].between(0, 1).all()


def test_frame_fingerprint_stable_and_sensitive(clean_wide):
    f1 = qio.frame_fingerprint(clean_wide)
    f2 = qio.frame_fingerprint(clean_wide.copy())
    assert f1 == f2 and len(f1) == 64
    modified = clean_wide.copy()
    modified.loc[modified.index[0], cfg.LABEL_COL] = 1 - int(modified.iloc[0][cfg.LABEL_COL])
    assert qio.frame_fingerprint(modified) != f1


def test_save_and_load_roundtrip_preserves_datetime_dtype(tmp_path, tables):
    """落盘再读回必须保持时间列是 datetime（否则「生成」与「读盘」两条路径结果不一致）。"""
    qio.save_raw_tables(tables, tmp_path)
    loaded = qio.load_raw_tables(tmp_path)
    assert pd.api.types.is_datetime64_any_dtype(loaded["production_runs"]["start_time"])
    assert pd.api.types.is_datetime64_any_dtype(loaded["inspection_results"]["inspection_ts"])
    for name in tables:
        pd.testing.assert_frame_equal(
            tables[name].reset_index(drop=True), loaded[name].reset_index(drop=True),
            check_dtype=False,
        )


def test_load_raw_tables_raises_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        qio.load_raw_tables(tmp_path)
