"""CLI 入口与端到端流程的测试。

覆盖：参数解析、各子命令、完整流程能否产出全部约定产物、
以及「连跑两次指标逐字一致」这一可复现性承诺。
"""

from __future__ import annotations

import hashlib
import json

import pytest

from qalab import cli
from qalab import config as cfg
from qalab import io as qio
from qalab.pipeline import ensure_raw_data, run_pipeline


# --------------------------------------------------------------------------
# 参数解析
# --------------------------------------------------------------------------
def test_build_parser_accepts_run_subcommand():
    args = cli.build_parser().parse_args(["run", "--out", "reports", "--seed", "7"])
    assert args.command == "run"
    assert str(args.out) == "reports"
    assert args.seed == 7
    assert args.n_perm == cfg.N_PERM


def test_build_parser_defaults():
    args = cli.build_parser().parse_args(["run"])
    assert args.out == cfg.DEFAULT_OUT_DIR
    assert args.data_dir == cfg.DEFAULT_DATA_DIR
    assert args.seed == cfg.SEED
    assert args.regenerate_data is False
    assert args.no_plots is False


def test_build_parser_stability_and_info():
    assert cli.build_parser().parse_args(["stability", "--seeds", "3"]).seeds == 3
    assert cli.build_parser().parse_args(["info"]).command == "info"


def test_build_parser_rejects_unknown_flag():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["run", "--not-a-flag"])


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert "qalab" in capsys.readouterr().out


def test_info_command_prints_config(capsys):
    code = cli.main(["info"])
    out = capsys.readouterr().out
    assert code == 0
    assert "pandas" in out and "scikit-learn" in out
    assert str(cfg.SEED) in out
    assert "furnace_temp_c" in out


# --------------------------------------------------------------------------
# 流程与产物
# --------------------------------------------------------------------------
def test_run_pipeline_writes_all_expected_artifacts(pipeline_result):
    out_dir = pipeline_result["paths"]["metrics"].parent
    expected = {
        "metrics.json",
        "实验报告.md",
        "factor_ranking.csv",
        "model_comparison.csv",
        "significance_tests.csv",
        "run_meta.json",
    }
    existing = {p.name for p in out_dir.iterdir()}
    assert expected.issubset(existing)

    pngs = sorted(p.name for p in out_dir.glob("*.png"))
    assert len(pngs) >= 9, f"至少应有 9 张图，实际 {pngs}"
    for name in (
        "01_label_balance.png",
        "02_numeric_distributions_by_label.png",
        "03_boxplots_by_label.png",
        "04_correlation_heatmap.png",
        "05_factor_importance.png",
        "06_confusion_matrix.png",
        "07_roc_pr_curves.png",
        "08_data_quality.png",
        "09_model_comparison.png",
    ):
        assert name in pngs
    # 图不能是空文件
    for path in out_dir.glob("*.png"):
        assert path.stat().st_size > 5000, f"{path.name} 太小，可能是空图"


def test_run_pipeline_metrics_content(pipeline_result):
    metrics = pipeline_result["metrics"]
    assert metrics["data"]["n_batches"] == cfg.N_BATCHES
    assert metrics["data"]["n_runs"] == cfg.N_BATCHES * cfg.RUNS_PER_BATCH
    assert metrics["data"]["n_wide_rows"] == metrics["data"]["n_inspections"]
    assert metrics["data"]["n_clean_rows"] <= metrics["data"]["n_wide_rows"]
    assert 0 < metrics["data"]["fail_rate"] < 1
    assert metrics["significance"]["n_factors_tested"] == len(cfg.MODEL_FEATURES)
    assert metrics["significance"]["n_significant"] == len(
        metrics["significance"]["significant_factors"]
    )
    assert metrics["quality"]["duplicates"]["n_full_duplicate_rows"] == cfg.N_INJECTED_DUPLICATE_ROWS
    assert metrics["cleaning"]["actions"]["rows_before"] == metrics["data"]["n_wide_rows"]
    assert len(metrics["data"]["clean_fingerprint"]) == 64


def test_pipeline_progress_log_is_emitted(tmp_path):
    lines: list[str] = []
    run_pipeline(
        out_dir=tmp_path / "out",
        data_dir=tmp_path / "data",
        make_plots=False,
        n_perm=50,
        log=lines.append,
    )
    assert len(lines) >= 8
    assert any("合并宽表" in line for line in lines)
    assert any("因子筛选" in line for line in lines)


def test_pipeline_outputs_are_byte_identical_across_two_runs(tmp_path):
    """可复现性硬承诺：同一命令连跑两次，除 run_meta 外全部产物逐字节一致。"""
    data_dir = tmp_path / "data"
    out1, out2 = tmp_path / "r1", tmp_path / "r2"
    run_pipeline(out_dir=out1, data_dir=data_dir, regenerate_data=True, n_perm=100, make_plots=False, log=lambda *a: None)
    run_pipeline(out_dir=out2, data_dir=data_dir, n_perm=100, make_plots=False, log=lambda *a: None)

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    for name in ("metrics.json", "实验报告.md", "factor_ranking.csv", "model_comparison.csv", "significance_tests.csv"):
        assert sha(out1 / name) == sha(out2 / name), f"{name} 两次运行不一致"
    # run_meta 含耗时与时间戳，应该不同
    assert sha(out1 / "run_meta.json") != sha(out2 / "run_meta.json")


def test_generate_then_load_path_gives_same_metrics(tmp_path):
    """「本次生成」与「读已有文件」两条路径必须给出完全相同的指标。

    开发中实测踩坑：内存里的 datetime64 与 CSV 读回的字符串不一致，
    导致数据指纹不同、可复现性检查失败。
    """
    data_dir = tmp_path / "data"
    a = run_pipeline(out_dir=tmp_path / "a", data_dir=data_dir, regenerate_data=True, n_perm=50, make_plots=False, log=lambda *a: None)
    b = run_pipeline(out_dir=tmp_path / "b", data_dir=data_dir, n_perm=50, make_plots=False, log=lambda *a: None)
    assert a["metrics"]["data"] == b["metrics"]["data"]
    assert a["metrics"]["significance"] == b["metrics"]["significance"]


def test_ensure_raw_data_regenerates_when_missing(tmp_path):
    data_dir = tmp_path / "empty"
    tables, manifest, generated = ensure_raw_data(data_dir, seed=cfg.SEED)
    assert generated is True
    assert manifest["constant_column"] == cfg.CONSTANT_COLUMN_NAME
    assert (data_dir / "material_batches.csv").exists()
    # 第二次调用应当直接读磁盘
    tables2, _manifest2, generated2 = ensure_raw_data(data_dir, seed=cfg.SEED)
    assert generated2 is False
    assert len(tables2["production_runs"]) == len(tables["production_runs"])


def test_ensure_raw_data_respects_regenerate_flag(tmp_path):
    data_dir = tmp_path / "d"
    ensure_raw_data(data_dir, seed=cfg.SEED)
    _t, _m, generated = ensure_raw_data(data_dir, seed=cfg.SEED, regenerate=True)
    assert generated is True


def test_cli_run_end_to_end(tmp_path, capsys):
    out_dir = tmp_path / "reports"
    code = cli.main(
        [
            "run",
            "--out",
            str(out_dir),
            "--data-dir",
            str(tmp_path / "data"),
            "--n-perm",
            "50",
            "--no-plots",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert (out_dir / "metrics.json").exists()
    assert (out_dir / "实验报告.md").exists()
    assert "因子重要性排序" in out
    assert "模型对比" in out
    assert "显著因子" in out

    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["significance"]["n_perm"] == 50


def test_cli_run_quiet_suppresses_log(tmp_path, capsys):
    code = cli.main(
        [
            "run",
            "--out",
            str(tmp_path / "r"),
            "--data-dir",
            str(tmp_path / "d"),
            "--n-perm",
            "50",
            "--no-plots",
            "--quiet",
        ]
    )
    assert code == 0
    assert capsys.readouterr().out.strip() == ""


def test_cli_stability_command(tmp_path, capsys):
    code = cli.main(
        ["stability", "--seeds", "2", "--out", str(tmp_path / "r"), "--n-perm", "50"]
    )
    out = capsys.readouterr().out
    assert code == 0
    payload = json.loads((tmp_path / "r" / "stability.json").read_text(encoding="utf-8"))
    assert payload["n_seeds"] == 2
    assert set(payload["recovery_counts"]) == set(cfg.MODEL_FEATURES)
    assert set(payload["primary_recovery"]) == set(
        payload["ground_truth"]["primary"]
    )
    assert "furnace_temp_c" in out


# --------------------------------------------------------------------------
# 完整流程的一致性（用 session fixture，避免重复跑）
# --------------------------------------------------------------------------
def test_pipeline_ranking_matches_metrics(pipeline_result):
    ranking = pipeline_result["ranking"]
    metrics = pipeline_result["metrics"]
    assert len(ranking) == metrics["significance"]["n_factors_tested"]
    assert sorted(ranking["rank"]) == list(range(1, len(ranking) + 1))
    sig_from_ranking = sorted(
        ranking.loc[ranking["factor_significant"], "factor"].tolist()
    )
    assert sig_from_ranking == sorted(metrics["significance"]["significant_factors"])


def test_pipeline_quality_matches_raw_tables(pipeline_result):
    q = pipeline_result["metrics"]["quality"]
    assert q["duplicates"]["n_full_duplicate_rows"] == cfg.N_INJECTED_DUPLICATE_ROWS
    assert cfg.CONSTANT_COLUMN_NAME in q["constant_columns"]
    assert q["class_balance"]["n"] == pipeline_result["metrics"]["data"]["n_wide_rows"]
    fan = q["perf_test_fanout"]
    assert fan["fanout_detected"] is True
    assert fan["max_tests_per_unit"] == 3


def test_pipeline_uses_clean_table_for_modeling(pipeline_result):
    """建模必须用清洗后的表：模型样本数应与清洗后行数一致。"""
    metrics = pipeline_result["metrics"]
    assert metrics["modeling"]["n_samples"] == metrics["data"]["n_clean_rows"]
    assert metrics["modeling"]["n_samples"] == len(pipeline_result["clean"])


def test_clean_table_has_no_missing_in_model_features(pipeline_result):
    clean = pipeline_result["clean"]
    for col in (*cfg.MODEL_NUMERIC, *cfg.MODEL_CATEGORICAL):
        assert not clean[col].isna().any(), f"{col} 清洗后仍存在缺失"


def test_run_level_and_batch_level_tables_are_consistent(pipeline_result):
    run_df = pipeline_result["run_level"]
    batch_df = pipeline_result["batch_level"]
    assert len(run_df) == cfg.N_BATCHES * cfg.RUNS_PER_BATCH
    assert len(batch_df) == cfg.N_BATCHES
    assert run_df["n_units"].sum() == len(pipeline_result["clean"])
    assert batch_df["n_units"].sum() == len(pipeline_result["clean"])
    assert abs(run_df["fail_rate"].mean() - batch_df["fail_rate"].mean()) < 0.05


def test_wide_table_shape_documented_in_metrics(pipeline_result):
    wide = pipeline_result["wide"]
    metrics = pipeline_result["metrics"]
    assert wide.shape == (metrics["data"]["n_wide_rows"], metrics["data"]["n_wide_columns"])
    assert qio.frame_fingerprint(wide) == metrics["data"]["wide_fingerprint"]
