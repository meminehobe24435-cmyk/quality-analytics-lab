"""配置与可视化模块的测试。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qalab import config as cfg
from qalab import plots


# --------------------------------------------------------------------------
# 配置自洽性
# --------------------------------------------------------------------------
def test_spec_limits_are_well_formed():
    for col, (low, high) in cfg.SPEC_LIMITS.items():
        assert isinstance(col, str) and col
        assert low < high, f"{col} 的上下限反了"


def test_column_groups_are_disjoint_and_cover_the_model_features():
    groups = {
        "production_numeric": set(cfg.PRODUCTION_NUMERIC),
        "material_numeric": set(cfg.MATERIAL_NUMERIC),
        "production_categorical": set(cfg.PRODUCTION_CATEGORICAL),
        "perf_numeric": set(cfg.PERF_NUMERIC),
    }
    names = list(groups)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            assert not (groups[a] & groups[b]), f"{a} 与 {b} 有重叠列"
    assert set(cfg.MODEL_FEATURES) == (
        set(cfg.PRODUCTION_NUMERIC)
        | set(cfg.MATERIAL_NUMERIC)
        | set(cfg.PRODUCTION_CATEGORICAL)
    )
    assert set(cfg.MODEL_NUMERIC) == set(cfg.PRODUCTION_NUMERIC) | set(cfg.MATERIAL_NUMERIC)
    assert set(cfg.ALL_NUMERIC) == set(cfg.MODEL_NUMERIC) | set(cfg.PERF_NUMERIC)


def test_leaky_columns_are_not_model_features():
    for col in cfg.LEAKY_COLUMNS:
        assert col not in cfg.MODEL_FEATURES, f"{col} 是泄漏字段，不能进特征"


def test_every_numeric_column_has_spec_limits():
    for col in cfg.ALL_NUMERIC:
        assert col in cfg.SPEC_LIMITS, f"{col} 缺少物理上下限"


def test_analysis_parameters_are_sane():
    assert 0 < cfg.ALPHA < 1
    assert cfg.CV_FOLDS >= 2
    assert 0 < cfg.TEST_SIZE < 1 and 0 < cfg.VAL_SIZE < 1
    assert cfg.TEST_SIZE + cfg.VAL_SIZE * (1 - cfg.TEST_SIZE) < 0.5
    assert cfg.N_PERM >= 100
    assert 0 < cfg.DRIFT_BASELINE_FRACTION < 1
    assert cfg.DRIFT_PSI_THRESHOLD >= 0.1
    assert 0 < cfg.DRIFT_KS_STAT_THRESHOLD <= 1


def test_scale_constants_make_a_workable_dataset():
    n_runs = cfg.N_BATCHES * cfg.RUNS_PER_BATCH
    assert n_runs >= 100, "炉次数太少，因子筛选的功效不足"
    assert cfg.N_BATCHES >= 20, "批次太少，原料因子的检验没有意义"
    assert cfg.UNITS_PER_RUN_MIN < cfg.UNITS_PER_RUN_MAX
    assert 0 < cfg.PERF_TEST_UNIT_FRACTION < 1
    assert 0 < cfg.PERF_MISSING_RATE < 0.5
    assert set(cfg.PERF_REPEAT_CHOICES) >= {1}
    assert max(cfg.PERF_REPEAT_CHOICES) >= 2, "需要复测才能制造一对多关系"


def test_raw_files_mapping_is_complete():
    assert set(cfg.RAW_FILES) == {
        "material_batches",
        "production_runs",
        "inspection_results",
        "perf_tests",
    }
    assert all(name.endswith(".csv") for name in cfg.RAW_FILES.values())


def test_package_paths_point_into_the_repo():
    assert cfg.PACKAGE_DIR.name == "qalab"
    assert cfg.PROJECT_ROOT == cfg.PACKAGE_DIR.parent
    assert isinstance(cfg.DEFAULT_OUT_DIR, Path)


def test_injected_anomaly_constants_are_positive():
    assert cfg.N_INJECTED_OUT_OF_RANGE > 0
    assert cfg.N_INJECTED_DUPLICATE_ROWS > 0
    assert cfg.N_INJECTED_MISSING_INSPECTOR > 0
    assert cfg.N_INJECTED_MISSING_MACHINE > 0
    assert cfg.N_INJECTED_OOS_PERF > 0
    assert 0 < cfg.NEAR_CONSTANT_TRUE_RATE < 0.05
    assert cfg.CONSTANT_COLUMN_NAME not in cfg.MODEL_FEATURES


# --------------------------------------------------------------------------
# 可视化
# --------------------------------------------------------------------------
def test_set_style_uses_non_interactive_backend():
    plots.set_style()
    import matplotlib

    assert matplotlib.get_backend().lower() == "agg"
    assert matplotlib.rcParams["savefig.dpi"] > 0


def test_plot_label_balance_writes_file(tmp_path, clean_wide):
    plots.set_style()
    path = plots.plot_label_balance(clean_wide, tmp_path)
    assert path.exists() and path.stat().st_size > 3000
    assert path.name == "01_label_balance.png"


def test_plot_numeric_distributions_writes_file(tmp_path, clean_wide):
    path = plots.plot_numeric_distributions_by_label(clean_wide, tmp_path)
    assert path.exists() and path.stat().st_size > 3000


def test_plot_boxplots_and_heatmap(tmp_path, clean_wide):
    p1 = plots.plot_boxplots_by_label(clean_wide, tmp_path)
    p2 = plots.plot_correlation_heatmap(clean_wide, tmp_path)
    for path in (p1, p2):
        assert path.exists() and path.stat().st_size > 3000


def test_plot_correlation_heatmap_handles_constant_column(tmp_path):
    """常量列进入相关性热图不应抛异常（Spearman 会出现 NaN 相关系数）。"""
    df = pd.DataFrame(
        {
            "furnace_temp_c": [1500.0, 1505.0, 1510.0, 1502.0],
            "const": [1.0, 1.0, 1.0, 1.0],
            "is_fail": [0, 1, 0, 1],
        }
    )
    path = plots.plot_correlation_heatmap(df, tmp_path)
    assert path.exists()


def test_plot_data_quality_with_no_missing_or_oob(tmp_path):
    summary = {"n_rows": 10, "constant_columns": [], "near_constant_columns": []}
    missing = pd.DataFrame({"column": ["a"], "n_missing": [0], "pct_missing": [0.0]})
    oob = pd.DataFrame({"column": ["a"], "n_below": [0], "n_above": [0], "n_out_of_range": [0]})
    path = plots.plot_data_quality(summary, missing, oob, tmp_path)
    assert path.exists()


def test_plot_confusion_matrix(tmp_path):
    path = plots.plot_confusion_matrix([[100, 5], [20, 30]], tmp_path, "logistic_regression")
    assert path.exists() and path.stat().st_size > 3000


def test_plot_roc_pr_curves(tmp_path):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    proba = {"m1": rng.random(200), "m2": rng.random(200)}
    scores = {
        "m1": {"average_precision": 0.5, "roc_auc": 0.7},
        "m2": {"average_precision": 0.4, "roc_auc": 0.6},
    }
    path = plots.plot_roc_pr_curves(y, proba, tmp_path, scores)
    assert path.exists() and path.stat().st_size > 3000


def test_plot_all_creates_nine_figures(tmp_path, pipeline_result):
    """plot_all 必须产出约定的 9 张图，并且在重复调用时覆盖而不是追加。"""
    out = tmp_path / "figs"
    paths = plots.plot_all(
        pipeline_result["clean"],
        pipeline_result["ranking"],
        pipeline_result["quality_summary"],
        pd.DataFrame(pipeline_result["quality_report"]["missing_by_column"]),
        pd.DataFrame(pipeline_result["quality_report"]["out_of_range"]),
        pipeline_result["modeling"],
        pipeline_result["modeling_extras"],
        out,
    )
    assert len(paths) == 9
    assert all(Path(p).exists() for p in paths)
    assert len(list(out.glob("*.png"))) == 9
    names = sorted(Path(p).name for p in paths)
    assert names == sorted(names)
    assert names[0].startswith("01_") and names[-1].startswith("09_")
