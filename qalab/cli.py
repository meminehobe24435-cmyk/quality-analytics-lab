"""命令行入口。

用法：
    py -3.12 -m qalab run --out reports          # 跑完整流程
    py -3.12 -m qalab run --regenerate-data      # 重新生成合成数据后再跑
    py -3.12 -m qalab stability --seeds 5        # 跨种子稳定性检查
    py -3.12 -m qalab info                       # 打印配置与依赖版本

Windows 控制台默认是 GBK，直接 print 中文可能抛 UnicodeEncodeError，
因此这里显式把 stdout/stderr 重配为 UTF-8。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import config as cfg
from .console import force_utf8_stdout


def _force_utf8_stdout() -> None:
    """兼容旧调用点（真正的实现已收敛到 qalab.console）。"""
    force_utf8_stdout()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qalab",
        description="质量数据分析工作台：多源合并 -> 数据质量检查 -> 显著因子识别 -> 建模 -> 报告",
    )
    parser.add_argument("--version", action="version", version=f"qalab {_version()}")
    sub = parser.add_subparsers(dest="command")

    p_run = sub.add_parser("run", help="跑完整分析流程")
    p_run.add_argument("--out", type=Path, default=cfg.DEFAULT_OUT_DIR, help="产物输出目录")
    p_run.add_argument(
        "--data-dir", type=Path, default=cfg.DEFAULT_DATA_DIR, help="原始数据目录"
    )
    p_run.add_argument("--seed", type=int, default=cfg.SEED, help="随机种子")
    p_run.add_argument(
        "--regenerate-data", action="store_true", help="强制重新生成合成数据"
    )
    p_run.add_argument("--no-plots", action="store_true", help="跳过绘图（快速冒烟）")
    p_run.add_argument(
        "--n-perm", type=int, default=cfg.N_PERM, help="簇置换检验的置换次数"
    )
    p_run.add_argument("--quiet", action="store_true", help="不打印过程日志")

    p_stab = sub.add_parser("stability", help="跨种子稳定性检查（主因子是否稳定被识别）")
    p_stab.add_argument("--seeds", type=int, default=5, help="种子个数")
    p_stab.add_argument("--out", type=Path, default=cfg.DEFAULT_OUT_DIR, help="输出目录")
    p_stab.add_argument("--n-perm", type=int, default=200, help="置换次数（稳定性检查用较少次数）")

    sub.add_parser("info", help="打印配置与依赖版本")
    return parser


def _version() -> str:
    from . import __version__

    return __version__


def _print_versions() -> None:
    import matplotlib
    import numpy
    import pandas
    import scipy
    import seaborn
    import sklearn

    print("依赖版本：")
    for name, mod in (
        ("pandas", pandas),
        ("numpy", numpy),
        ("matplotlib", matplotlib),
        ("seaborn", seaborn),
        ("scikit-learn", sklearn),
        ("scipy", scipy),
    ):
        print(f"  {name:14s} {mod.__version__}")


def cmd_info(_args: argparse.Namespace) -> int:
    _print_versions()
    print(f"\nPython {sys.version.split()[0]}")
    print(f"全局种子 SEED = {cfg.SEED}")
    print(f"规模：{cfg.N_BATCHES} 批次 × {cfg.RUNS_PER_BATCH} 炉次 = "
          f"{cfg.N_BATCHES * cfg.RUNS_PER_BATCH} 个 run")
    print(f"显著性水平 ALPHA = {cfg.ALPHA}，交叉验证折数 = {cfg.CV_FOLDS}")
    print(f"置换次数 N_PERM = {cfg.N_PERM}")
    print("\n列分组：")
    print(f"  生产数值 {list(cfg.PRODUCTION_NUMERIC)}")
    print(f"  原料数值 {list(cfg.MATERIAL_NUMERIC)}")
    print(f"  生产类别 {list(cfg.PRODUCTION_CATEGORICAL)}")
    print(f"  性能测试 {list(cfg.PERF_NUMERIC)}")
    print(f"  建模剔除（泄漏字段） {list(cfg.LEAKY_COLUMNS)}")
    print("\n物理/规格上下限：")
    for col, (low, high) in cfg.SPEC_LIMITS.items():
        print(f"  {col:24s} [{low}, {high}]")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from .pipeline import run_pipeline

    log = (lambda *a, **k: None) if args.quiet else print
    result = run_pipeline(
        out_dir=args.out,
        data_dir=args.data_dir,
        seed=args.seed,
        regenerate_data=args.regenerate_data,
        make_plots=not args.no_plots,
        n_perm=args.n_perm,
        log=log,
    )

    ranking = result["ranking"]
    metrics = result["metrics"]
    log("")
    log("=== 因子重要性排序（前 8）===")
    cols = ["rank", "factor", "factor_level", "factor_p_value", "factor_q_value", "verdict"]
    log(ranking[cols].head(8).to_string(index=False))
    log("")
    log("=== 模型对比（测试集，只评估一次）===")
    rows = []
    for name, m in metrics["modeling"]["models"].items():
        tm = m["test_metrics"]
        rows.append(
            {
                "model": name,
                "threshold": round(m["threshold_selection"]["threshold"], 3),
                "AP": round(tm["average_precision"], 4),
                "ROC_AUC": round(tm["roc_auc"], 4),
                "precision": round(tm["precision"], 4),
                "recall": round(tm["recall"], 4),
                "F1": round(tm["f1"], 4),
            }
        )
    import pandas as pd

    log(pd.DataFrame(rows).to_string(index=False))
    log("")
    log(f"显著因子（BH q<={cfg.ALPHA}）：{metrics['significance']['significant_factors']}")
    log(f"产物目录：{Path(args.out).resolve()}")
    return 0


def cmd_stability(args: argparse.Namespace) -> int:
    """跨种子稳定性：只跑 数据生成 + 质量检查 + 因子筛选（不建模），比较结论是否稳定。"""
    import pandas as pd

    from . import io as qio
    from . import significance as qsig
    from . import synth
    from .report import write_json

    manifest_truth: dict[str, list[str]] = {}
    rows = []
    for i in range(args.seeds):
        seed = cfg.SEED + i * 1000
        tables, manifest = synth.generate_all(seed)
        if not manifest_truth:
            truth = manifest["ground_truth_mechanism"]
            manifest_truth = {
                "primary": list(truth["primary_factors"]),
                "decoy": list(truth["decoy_factors"]),
            }
        wide = qio.build_analysis_table(tables)
        clean, _ = qio.clean_analysis_table(wide)
        tests = qsig.screen_factors(
            clean,
            qio.build_run_level_table(clean),
            qio.build_batch_level_table(clean),
            seed=seed,
            n_perm=args.n_perm,
        )
        ranking = qsig.rank_factors(tests)
        ranking = ranking.assign(seed=seed)
        rows.append(ranking)
        print(f"  seed={seed} 显著因子 {qsig.significant_factors(ranking, 'factor')}")

    allr = pd.concat(rows, ignore_index=True)
    summary = (
        allr.groupby("factor")[
            ["unit_naive_significant", "unit_significant", "factor_significant"]
        ]
        .sum()
        .astype(int)
        .reset_index()
    )
    payload = {
        "n_seeds": args.seeds,
        "seeds": [cfg.SEED + i * 1000 for i in range(args.seeds)],
        "n_perm": args.n_perm,
        "ground_truth": manifest_truth,
        "recovery_counts": {
            row["factor"]: {
                "naive_unit": int(row["unit_naive_significant"]),
                "cluster_permutation": int(row["unit_significant"]),
                "factor_level": int(row["factor_significant"]),
            }
            for _, row in summary.iterrows()
        },
        "primary_recovery": {
            f: int(summary.loc[summary["factor"] == f, "factor_significant"].iloc[0])
            for f in manifest_truth.get("primary", [])
            if f in set(summary["factor"])
        },
        "decoy_false_positives": {
            f: int(summary.loc[summary["factor"] == f, "factor_significant"].iloc[0])
            for f in manifest_truth.get("decoy", [])
            if f in set(summary["factor"])
        },
    }
    out = Path(args.out) / "stability.json"
    write_json(payload, out)
    print("")
    print(summary.to_string(index=False))
    print(f"\n已写出 {out}")
    print(
        "主因子在不同种子下被识别的次数："
        + json.dumps(payload["primary_recovery"], ensure_ascii=False)
    )
    print(
        "真值零效应因子被误判的次数："
        + json.dumps(payload["decoy_false_positives"], ensure_ascii=False)
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    _force_utf8_stdout()
    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command or "run"
    if args.command is None:
        # 不带子命令时按 run 执行，保持 `py -3.12 -m qalab --out reports` 也能用
        args = parser.parse_args(["run"])
    if command == "run":
        return cmd_run(args)
    if command == "stability":
        return cmd_stability(args)
    if command == "info":
        return cmd_info(args)
    parser.error(f"未知命令: {command}")  # pragma: no cover
    return 2  # pragma: no cover
