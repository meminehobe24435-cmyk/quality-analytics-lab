"""报告生成与产物的测试。

覆盖：metrics.json 的结构与可序列化、Markdown 报告包含必需章节与真实数字、
CSV 明细导出、run_meta 与 metrics 的职责分离（前者含耗时、后者不含）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from qalab import config as cfg
from qalab import report as r


def test_write_json_roundtrip(tmp_path):
    path = r.write_json({"a": 1, "b": [1, 2], "中文": "值"}, tmp_path / "x.json")
    assert path.exists()
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded == {"a": 1, "b": [1, 2], "中文": "值"}
    # 末尾应当有换行（便于 git diff 与 cat）
    assert path.read_text(encoding="utf-8").endswith("\n")


def test_write_csv_creates_file(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    path = r.write_csv(df, tmp_path / "sub" / "t.csv")
    assert path.exists()
    back = pd.read_csv(path)
    assert back.shape == df.shape


def test_write_run_meta_contains_versions_and_elapsed(tmp_path):
    path = r.write_run_meta(
        tmp_path / "run_meta.json",
        tmp_path,
        {"a": 1.23456},
        seed=cfg.SEED,
        extra={"data_source": "测试"},
    )
    meta = json.loads(path.read_text(encoding="utf-8"))
    assert meta["seed"] == cfg.SEED
    assert set(meta["versions"]) >= {"pandas", "numpy", "matplotlib", "seaborn", "scikit-learn", "scipy"}
    assert meta["elapsed_seconds"]["a"] == pytest.approx(1.2346, abs=1e-3)
    assert meta["data_source"] == "测试"
    assert "generated_at_utc" in meta
    assert "不参与" in meta["note"]


def test_formatting_helpers():
    assert r._fmt(1.23456) == "1.2346"
    assert r._fmt(None) == "n/a"
    assert r._fmt(float("nan")) == "n/a"
    assert r._fmt(3) == "3"
    assert r._fmt_p(0.0000123) == "1.23e-05"
    assert r._fmt_p(0.25) == "0.2500"
    assert r._fmt_p(None) == "n/a"
    assert r._fmt_p(0.0) == "<1e-300"


def test_table_builder():
    text = r._table(["A", "B"], [["1", "2"], ["3", "4"]])
    lines = text.splitlines()
    assert lines[0] == "| A | B |"
    assert lines[1] == "|---|---|"
    assert lines[2] == "| 1 | 2 |"


def test_build_report_markdown_has_required_sections(pipeline_result):
    metrics = pipeline_result["metrics"]
    ranking = pipeline_result["ranking"]
    markdown = r.build_report_markdown(metrics, ranking, pipeline_result["plots"])
    for section in (
        "# 质量数据分析实验报告",
        "## 1. 数据规模与合并结果",
        "## 2. 数据质量检查结论",
        "## 3. 显著因子识别",
        "## 4. 统计检验",
        "## 5. 预测模型对比",
        "## 6. 图表",
        "## 7. 可复现性与边界",
    ):
        assert section in markdown, f"报告缺少章节 {section}"
    # 必须写明这是合成数据
    assert "合成" in markdown
    assert "不是任何真实产线" in markdown


def test_report_contains_real_numbers_from_metrics(pipeline_result):
    """报告里的数字必须来自 metrics，而不是写死的文案。"""
    metrics = pipeline_result["metrics"]
    ranking = pipeline_result["ranking"]
    markdown = r.build_report_markdown(metrics, ranking, pipeline_result["plots"])

    assert str(metrics["data"]["n_runs"]) in markdown
    assert str(metrics["data"]["n_wide_rows"]) in markdown
    assert metrics["data"]["clean_fingerprint"][:16] in markdown
    for factor in metrics["significance"]["significant_factors"]:
        assert factor in markdown
    for name in metrics["modeling"]["models"]:
        assert name in markdown
    # 每个显著因子的 q 值格式必须出现在报告里
    assert f"{ranking.iloc[0]['factor_q_value']:.2e}" in markdown


def test_report_handles_empty_ranking(pipeline_result):
    """排名为空（例如所有因子都不显著）时报告仍要能生成。"""
    metrics = pipeline_result["metrics"]
    markdown = r.build_report_markdown(metrics, pd.DataFrame(), [])
    assert "# 质量数据分析实验报告" in markdown
    assert "显著因子" in markdown


def test_report_lists_plot_filenames(pipeline_result):
    metrics = pipeline_result["metrics"]
    plots = pipeline_result["plots"]
    markdown = r.build_report_markdown(metrics, pipeline_result["ranking"], plots)
    for path in plots:
        assert Path(path).name in markdown


def test_metrics_json_is_json_and_has_all_top_level_sections(pipeline_result):
    metrics_path = pipeline_result["paths"]["metrics"]
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    for key in ("run", "data", "quality", "cleaning", "significance", "modeling"):
        assert key in payload
    assert payload["run"]["seed"] == cfg.SEED
    assert payload["significance"]["alpha"] == cfg.ALPHA
    assert payload["data"]["n_fail"] > 0


def test_metrics_does_not_contain_timing_or_timestamp(pipeline_result):
    """metrics.json 必须不含耗时/时间戳，否则无法逐字节比对（可复现性）。"""
    text = pipeline_result["paths"]["metrics"].read_text(encoding="utf-8")
    for forbidden in ("elapsed", "generated_at", "seconds", "platform", "python"):
        assert forbidden not in text, f"metrics.json 不应包含 {forbidden}"
    meta = json.loads(pipeline_result["paths"]["run_meta"].read_text(encoding="utf-8"))
    assert "elapsed_seconds" in meta
    assert "generated_at_utc" in meta


def test_csv_artifacts_have_expected_columns(pipeline_result):
    ranking = pd.read_csv(pipeline_result["paths"]["factor_ranking"], encoding="utf-8")
    assert {"rank", "factor", "factor_p_value", "factor_q_value", "verdict"}.issubset(
        ranking.columns
    )
    assert len(ranking) == pipeline_result["metrics"]["significance"]["n_factors_tested"]

    comparison = pd.read_csv(pipeline_result["paths"]["model_comparison"], encoding="utf-8")
    assert {"model", "ap", "roc_auc", "f1"}.issubset(comparison.columns)
    assert len(comparison) == 3

    tests = pd.read_csv(pipeline_result["paths"]["significance_tests"], encoding="utf-8")
    assert {"level", "factor", "method", "p_value", "q_value"}.issubset(tests.columns)
    assert tests["level"].nunique() == 3


def test_report_file_is_utf8_and_written(pipeline_result):
    report_path = pipeline_result["paths"]["report"]
    assert report_path.exists()
    assert report_path.name == "实验报告.md"
    text = report_path.read_text(encoding="utf-8")
    assert "质量数据分析实验报告" in text
