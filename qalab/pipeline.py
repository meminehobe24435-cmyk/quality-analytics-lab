"""端到端流程编排：数据生成/读取 -> 合并 -> 质量检查 -> 清洗 -> 因子筛选
-> 统计检验 -> 建模 -> 出图 -> 报告。

把编排放在这里（而不是 CLI 里），是为了让测试可以直接调用流程、
不必去解析命令行输出。
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg
from . import cluster as qcluster
from . import io as qio
from . import model as qmodel
from . import plots as qplots
from . import quality as qquality
from . import regression as qregression
from . import report as qreport
from . import significance as qsig
from . import synth
from . import timeseries as qtimeseries


def ensure_raw_data(
    data_dir: Path, seed: int = cfg.SEED, regenerate: bool = False
) -> tuple[dict[str, pd.DataFrame], dict[str, object], bool]:
    """准备原始数据：存在就直接读，不存在（或要求重新生成）就按固定种子生成。

    返回 (tables, manifest, generated)。

    ⚠️ 生成之后**也要重新从磁盘读一遍**：这样无论本次是「刚生成」还是「读已有文件」，
    进入分析的 DataFrame 都是同一套由 CSV 解析出来的 dtype。
    实测踩坑：不重读时，内存里是 datetime64、磁盘上是字符串，
    两个路径算出的「数据指纹」不同，直接把可复现性比对搞失败。
    """
    data_dir = Path(data_dir)
    import json

    manifest_path = data_dir / "synthetic_manifest.json"
    have_all = all((data_dir / f).exists() for f in cfg.RAW_FILES.values())

    generated = False
    if regenerate or not have_all:
        tables, manifest = synth.generate_all(seed)
        qio.save_raw_tables(tables, data_dir)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        generated = True

    tables = qio.load_raw_tables(data_dir)
    manifest = {}
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as fh:
            manifest = json.load(fh)
    return tables, manifest, generated



def _top_factor_detail(tests: pd.DataFrame, ranking: pd.DataFrame) -> dict[str, object]:
    """把排名第一的因子的各口径检验结果整理成一段便于报告的明细。"""
    if ranking.empty:
        return {}
    factor = ranking.iloc[0]["factor"]
    sub = tests[tests["factor"] == factor]

    def pick(level: str, role: str):
        sel = sub[(sub["level"] == level) & (sub["role"] == role)]
        return sel.iloc[0] if len(sel) else None

    own = pick(qsig.LEVEL_FACTOR, qsig.ROLE_PRIMARY)
    cluster = pick(qsig.LEVEL_CLUSTER, qsig.ROLE_PRIMARY)
    naive = pick(qsig.LEVEL_NAIVE, qsig.ROLE_PRIMARY)
    supp = pick(qsig.LEVEL_NAIVE, qsig.ROLE_SUPPLEMENTARY)

    def stat(row, key):
        return None if row is None else row[key]

    return {
        "factor": factor,
        "primary_method": stat(own, "method") or "n/a",
        "primary_statistic": stat(own, "statistic"),
        "primary_p_value": stat(own, "p_value"),
        "effect_size": stat(own, "effect_size"),
        "effect_metric": stat(own, "effect_metric") or "n/a",
        "cluster_method": stat(cluster, "method") or "n/a",
        "cluster_statistic": stat(cluster, "statistic"),
        "cluster_p_value": stat(cluster, "p_value"),
        "naive_method": stat(naive, "method") or "n/a",
        "naive_statistic": stat(naive, "statistic"),
        "naive_p_value": stat(naive, "p_value"),
        "supplementary_method": stat(supp, "method"),
        "supplementary_p_value": stat(supp, "p_value"),
    }


def _clean_float(value):
    """把 numpy 标量转成 Python 标量，保证 JSON 可序列化。"""
    if value is None:
        return None
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and (np.isnan(value)):
        return None
    return value


def run_pipeline(
    out_dir: Path = cfg.DEFAULT_OUT_DIR,
    data_dir: Path = cfg.DEFAULT_DATA_DIR,
    seed: int = cfg.SEED,
    regenerate_data: bool = False,
    make_plots: bool = True,
    n_perm: int = cfg.N_PERM,
    log=print,
) -> dict[str, object]:
    """跑完整流程，写出 metrics.json / run_meta.json / 实验报告与图表。

    返回一个包含全部产物的字典（同时是 metrics.json 的内容 + 内存对象）。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    tables, manifest, generated = ensure_raw_data(data_dir, seed, regenerate_data)
    timings["ensure_data"] = time.perf_counter() - t0
    log(f"[1/8] 原始数据{'已生成' if generated else '已读取'}: "
        + ", ".join(f"{k}={v.shape}" for k, v in tables.items()))

    t0 = time.perf_counter()
    wide = qio.build_analysis_table(tables)
    timings["merge"] = time.perf_counter() - t0
    log(f"[2/8] 合并宽表: {wide.shape[0]} 行 × {wide.shape[1]} 列")

    t0 = time.perf_counter()
    quality_report = qquality.run_quality_checks(
        wide, perf_tests=tables["perf_tests"]
    )
    quality_summary = qquality.summarize_quality(quality_report)
    timings["quality"] = time.perf_counter() - t0
    log(f"[3/8] 数据质量: 越界 {quality_summary['total_out_of_range_values']} 个, "
        f"重复行 {quality_summary['full_duplicate_rows']}, "
        f"常量列 {quality_summary['constant_columns']}, "
        f"漂移列 {quality_summary['drifted_columns']}")

    t0 = time.perf_counter()
    clean, clean_actions = qio.clean_analysis_table(wide)
    timings["clean"] = time.perf_counter() - t0
    log(f"[4/8] 清洗: {clean_actions['rows_before']} -> {clean_actions['rows_final']} 行")

    t0 = time.perf_counter()
    run_df = qio.build_run_level_table(clean)
    batch_df = qio.build_batch_level_table(clean)
    tests = qsig.screen_factors(clean, run_df, batch_df, seed=seed, n_perm=n_perm)
    ranking = qsig.rank_factors(tests)
    timings["significance"] = time.perf_counter() - t0
    log(f"[5/8] 因子筛选: 检验 {len(ranking)} 个因子, "
        f"显著 {int(ranking['factor_significant'].sum())} 个")

    t0 = time.perf_counter()
    modeling, extras = qmodel.run_modeling(clean, seed=seed)
    timings["modeling"] = time.perf_counter() - t0
    log(f"[6/12] 建模: 主模型 {modeling['best_model_by_validation_ap']}, "
        f"测试集 AP={modeling['models'][modeling['best_model_by_validation_ap']]['test_metrics']['average_precision']:.4f}, "
        f"共 {len(modeling['models'])} 个模型（含神经网络）")

    t0 = time.perf_counter()
    clustering, cluster_extras = qcluster.run_clustering(clean, seed=seed)
    timings["clustering"] = time.perf_counter() - t0
    log(f"[7/12] 聚类: 不合格样本 {clustering['n_samples']} 个 -> k={clustering['k_selected']}, "
        f"轮廓系数={clustering['silhouette']}, 簇规模={clustering['cluster_sizes']}")

    t0 = time.perf_counter()
    regression, regression_extras = qregression.run_regression(clean, seed=seed)
    timings["regression"] = time.perf_counter() - t0
    log(f"[8/12] 回归: 主目标 {regression['primary_target']} -> {regression.get('note', '')}")

    t0 = time.perf_counter()
    timeseries_result, timeseries_extras = qtimeseries.run_timeseries(
        clean, value_cols=(cfg.PERF_NUMERIC[0], cfg.PERF_NUMERIC[1])
    )
    timings["timeseries"] = time.perf_counter() - t0
    log(f"[9/12] 时间序列: {timeseries_result.get('trend_summary', {})}")

    t0 = time.perf_counter()
    plot_paths: list[Path] = []
    if make_plots:
        plot_paths = qplots.plot_all(
            clean,
            ranking,
            quality_summary,
            pd.DataFrame(quality_report["missing_by_column"]),
            pd.DataFrame(quality_report["out_of_range"]),
            modeling,
            extras,
            out_dir,
            extra_analyses={
                "clustering": (clustering, cluster_extras),
                "regression": (regression, regression_extras),
                "timeseries": (timeseries_result, timeseries_extras),
            },
        )
    timings["plots"] = time.perf_counter() - t0
    log(f"[10/12] 图表: {len(plot_paths)} 张")

    # ---------------- 组装 metrics（可复算：无耗时/时间戳） ----------------
    naive_sig = ranking.loc[ranking["unit_naive_significant"], "factor"].tolist()
    naive_fp = ranking.loc[
        ranking["unit_naive_significant"] & ~ranking["factor_significant"], "factor"
    ].tolist()
    significant = qsig.significant_factors(ranking, "factor")

    metrics: dict[str, object] = {
        "run": {
            "seed": int(seed),
            "alpha": cfg.ALPHA,
            "cv_folds": cfg.CV_FOLDS,
            "n_perm": int(n_perm),
        },
        "data": {
            "n_batches": int(len(tables["material_batches"])),
            "n_runs": int(len(tables["production_runs"])),
            "n_inspections": int(len(tables["inspection_results"])),
            "n_perf_tests": int(len(tables["perf_tests"])),
            "n_wide_rows": int(len(wide)),
            "n_wide_columns": int(wide.shape[1]),
            "n_clean_rows": int(len(clean)),
            "n_fail": int(clean[cfg.LABEL_COL].sum()),
            "fail_rate": float(clean[cfg.LABEL_COL].mean()),
            "perf_fanout_max": int(
                tables["perf_tests"].groupby("unit_id").size().max()
            ),
            "wide_fingerprint": qio.frame_fingerprint(wide),
            "clean_fingerprint": qio.frame_fingerprint(clean),
            "synthetic_manifest": manifest,
        },
        "quality": {
            **quality_summary,
            "missing_by_row": quality_report["missing_by_row"],
            "oob_columns": sorted(
                {r["column"] for r in quality_report["out_of_range"] if r["n_out_of_range"]}
            ),
            "perf_test_fanout": quality_report.get("perf_test_fanout", {}),
            "drift_detail": quality_report["drift_vs_baseline"],
        },
        "cleaning": {"actions": clean_actions},
        "significance": {
            "n_factors_tested": int(len(ranking)),
            "alpha": cfg.ALPHA,
            "n_perm": int(n_perm),
            "perm_p_floor": 1.0 / (n_perm + 1),
            "n_batches": int(len(batch_df)),
            "n_runs": int(len(run_df)),
            "significant_factors": significant,
            "n_significant": len(significant),
            "naive_significant_factors": naive_sig,
            "n_naive_significant": len(naive_sig),
            "naive_only_significant": [
                f for f in naive_sig if f not in significant
            ],
            "n_naive_only_significant": len([f for f in naive_sig if f not in significant]),
            "naive_false_positives": naive_fp,
            "n_naive_false_positive": len(naive_fp),
            "top_factor_detail": _top_factor_detail(tests, ranking),
        },
        "modeling": modeling,
        "clustering": clustering,
        "regression": regression,
        "timeseries": timeseries_result,
    }

    t0 = time.perf_counter()
    metrics_path = qreport.write_json(metrics, out_dir / "metrics.json")
    ranking_path = qreport.write_csv(ranking, out_dir / "factor_ranking.csv")
    cluster_profile_path = qreport.write_csv(
        cluster_extras.get("profile", pd.DataFrame()), out_dir / "cluster_profile.csv"
    )
    comparison = pd.DataFrame(
        [
            {
                "model": name,
                "threshold": m["threshold_selection"]["threshold"],
                "ap": m["test_metrics"]["average_precision"],
                "roc_auc": m["test_metrics"]["roc_auc"],
                "precision": m["test_metrics"]["precision"],
                "recall": m["test_metrics"]["recall"],
                "f1": m["test_metrics"]["f1"],
                "accuracy": m["test_metrics"]["accuracy"],
                "cv_ap_mean": metrics["modeling"]["cv"][name]["average_precision"]["mean"],
                "cv_ap_std": metrics["modeling"]["cv"][name]["average_precision"]["std"],
            }
            for name, m in metrics["modeling"]["models"].items()
        ]
    )
    comparison_path = qreport.write_csv(comparison, out_dir / "model_comparison.csv")
    tests_path = qreport.write_csv(tests, out_dir / "significance_tests.csv")

    markdown = qreport.build_report_markdown(metrics, ranking, plot_paths)
    report_path = out_dir / "实验报告.md"
    report_path.write_text(markdown, encoding="utf-8")
    timings["report"] = time.perf_counter() - t0

    meta_path = qreport.write_run_meta(
        out_dir / "run_meta.json",
        out_dir,
        timings,
        seed,
        extra={"data_source": "本次运行重新生成并落盘" if generated else "从磁盘读取已有文件"},
    )
    log(f"[11/12] 产物: {metrics_path.name}, {report_path.name}, "
        f"{ranking_path.name}, {comparison_path.name}, {tests_path.name}, "
        f"{cluster_profile_path.name}, {meta_path.name}, {len(plot_paths)} 张图")

    return {
        "metrics": metrics,
        "ranking": ranking,
        "tests": tests,
        "wide": wide,
        "clean": clean,
        "run_level": run_df,
        "batch_level": batch_df,
        "quality_report": quality_report,
        "quality_summary": quality_summary,
        "modeling": modeling,
        "modeling_extras": extras,
        "plots": plot_paths,
        "paths": {
            "metrics": metrics_path,
            "report": report_path,
            "factor_ranking": ranking_path,
            "model_comparison": comparison_path,
            "significance_tests": tests_path,
            "cluster_profile": cluster_profile_path,
            "run_meta": meta_path,
        },
        "clustering": clustering,
        "cluster_extras": cluster_extras,
        "regression": regression,
        "regression_extras": regression_extras,
        "timeseries": timeseries_result,
        "timeseries_extras": timeseries_extras,
        "timings": timings,
    }
