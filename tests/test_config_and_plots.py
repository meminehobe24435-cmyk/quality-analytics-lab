"""配置与可视化模块的测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qalab import config as cfg
from qalab import console, plots


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
# 控制台编码（Windows 上踩过三次的坑）
# --------------------------------------------------------------------------
def test_force_utf8_stdout_is_safe_to_call(capsys):
    """重复调用不应抛异常，且之后仍能正常打印中文。"""
    console.force_utf8_stdout()
    console.force_utf8_stdout()
    print("中文输出测试 实验报告.md")
    out = capsys.readouterr().out
    assert "实验报告.md" in out


def test_force_utf8_stdout_tolerates_streams_without_reconfigure(monkeypatch):
    """对没有 reconfigure 的流（被替换过的 stdout）必须静默跳过而不是崩溃。"""

    class Dummy:
        def write(self, _text):  # pragma: no cover - 仅占位
            pass

    monkeypatch.setattr(sys, "stdout", Dummy())
    console.force_utf8_stdout()  # 不应抛异常


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


def test_plot_all_creates_all_figures_with_extra_analyses(tmp_path, pipeline_result):
    """plot_all 必须产出全部 15 张图（含聚类/回归/时序），并在重复调用时覆盖。"""
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
        extra_analyses={
            "clustering": (pipeline_result["clustering"], pipeline_result["cluster_extras"]),
            "regression": (pipeline_result["regression"], pipeline_result["regression_extras"]),
            "timeseries": (pipeline_result["timeseries"], pipeline_result["timeseries_extras"]),
        },
    )
    assert len(paths) == 15
    assert all(Path(p).exists() for p in paths)
    assert len(list(out.glob("*.png"))) == 15
    names = sorted(Path(p).name for p in paths)
    assert names[0].startswith("01_") and names[-1].startswith("15_")
    # 不带 extra_analyses 时保持向后兼容（只出前 9 张）
    basic = plots.plot_all(
        pipeline_result["clean"],
        pipeline_result["ranking"],
        pipeline_result["quality_summary"],
        pd.DataFrame(pipeline_result["quality_report"]["missing_by_column"]),
        pd.DataFrame(pipeline_result["quality_report"]["out_of_range"]),
        pipeline_result["modeling"],
        pipeline_result["modeling_extras"],
        tmp_path / "basic",
    )
    assert len(basic) == 9


def test_no_chinese_text_is_drawn_on_any_figure(tmp_path, pipeline_result, monkeypatch):
    """图里的所有文字必须是英文 —— CI 上没有中文字体，中文会变成方块。

    做法：把 `_save` 换成探针，在保存前遍历图上所有 Text 对象收集字符串，
    再断言一个 CJK 字符都没有。这样能真正覆盖到标题/坐标轴/图例/注释。
    """
    import re

    import matplotlib.text as mtext

    collected: list[str] = []
    original = plots._save

    def spy(fig, out_dir, name):
        for text_obj in fig.findobj(mtext.Text):
            collected.append(text_obj.get_text())
        return original(fig, out_dir, name)

    monkeypatch.setattr(plots, "_save", spy)
    plots.plot_all(
        pipeline_result["clean"],
        pipeline_result["ranking"],
        pipeline_result["quality_summary"],
        pd.DataFrame(pipeline_result["quality_report"]["missing_by_column"]),
        pd.DataFrame(pipeline_result["quality_report"]["out_of_range"]),
        pipeline_result["modeling"],
        pipeline_result["modeling_extras"],
        tmp_path / "figs",
        extra_analyses={
            "clustering": (pipeline_result["clustering"], pipeline_result["cluster_extras"]),
            "regression": (pipeline_result["regression"], pipeline_result["regression_extras"]),
            "timeseries": (pipeline_result["timeseries"], pipeline_result["timeseries_extras"]),
        },
    )
    assert len(collected) > 50, "没有采集到图上的文字，探针可能失效了"
    cjk = re.compile(r"[\u4e00-\u9fff]")
    offenders = sorted({t for t in collected if cjk.search(t)})
    assert offenders == [], f"图里出现了中文（CI 上会变成方块）: {offenders[:5]}"


def test_trend_label_translation():
    """时序趋势判定文案必须被翻译成英文，且未知取值原样透传。"""
    assert plots._trend_en("无显著趋势") == "no significant trend"
    assert plots._trend_en("显著上升趋势") == "significant upward trend"
    assert plots._trend_en("显著下降趋势") == "significant downward trend"
    assert plots._trend_en("无法判定") == "undetermined"
    assert plots._trend_en("something-else") == "something-else"
    assert plots._trend_en(None) == "None"
