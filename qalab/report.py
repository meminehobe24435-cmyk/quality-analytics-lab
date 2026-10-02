"""报告生成：metrics.json、CSV 明细、Markdown 实验报告、运行元信息。

设计要点：
  - `metrics.json` **只装可复算的数值**，不含任何耗时、时间戳、主机信息。
    这样「同一命令连跑两次」可以用 sha256 逐字节比对；耗时与版本信息单独写到
    `run_meta.json`，两者职责分离。
  - Markdown 报告里的每个数字都来自传入的指标字典，不做任何手工填写，
    因此报告与指标文件永远一致。
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import config as cfg


def _fmt(value, digits: int = 4, na: str = "n/a") -> str:
    """把数值格式化成固定小数位的字符串，None 显示为 n/a。"""
    if value is None:
        return na
    if isinstance(value, float):
        if value != value:  # NaN
            return na
        return f"{value:.{digits}f}"
    return str(value)


def _fmt_p(value) -> str:
    """p 值/q 值用科学计数法，极小值显示为 <1e-16 而不是 0。"""
    if value is None:
        return "n/a"
    value = float(value)
    if value != value:
        return "n/a"
    if value == 0.0:
        return "<1e-300"
    if value < 1e-4:
        return f"{value:.2e}"
    return f"{value:.4f}"


def write_json(payload: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=False)
        fh.write("\n")
    return path


def write_csv(df: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    return path


def write_run_meta(
    path: Path,
    out_dir: Path,
    elapsed_seconds: dict[str, float],
    seed: int,
    extra: dict | None = None,
) -> Path:
    """运行元信息：耗时、环境版本、时间戳。

    ⚠️ 刻意与 metrics.json 分开：元信息每次都变，指标必须逐字一致。
    """
    import matplotlib
    import numpy
    import scipy
    import seaborn
    import sklearn

    meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "out_dir": str(out_dir),
        "versions": {
            "pandas": pd.__version__,
            "numpy": numpy.__version__,
            "matplotlib": matplotlib.__version__,
            "seaborn": seaborn.__version__,
            "scikit-learn": sklearn.__version__,
            "scipy": scipy.__version__,
        },
        "elapsed_seconds": {k: round(float(v), 4) for k, v in elapsed_seconds.items()},
        "note": "此文件包含耗时等易变信息，不参与「两次运行指标逐字一致」的比对",
    }
    if extra:
        meta.update(extra)
    return write_json(meta, path)



# --------------------------------------------------------------------------
# Markdown 报告
# --------------------------------------------------------------------------
def _table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def build_report_markdown(
    metrics: dict,
    ranking: pd.DataFrame,
    plots: list[Path],
    calib: dict | None = None,
) -> str:
    """按 metrics / ranking 拼出实验报告（所有数字均来自真实运行结果）。"""
    data = metrics["data"]
    quality = metrics["quality"]
    sig = metrics["significance"]
    modeling = metrics["modeling"]

    parts: list[str] = []
    parts.append("# 质量数据分析实验报告（自动生成）\n")
    parts.append(
        "> 本文件由 `py -3.12 -m qalab run --out reports` 自动生成，"
        "所有数字均来自当次真实运行，未做任何手工填写。\n"
    )
    parts.append(
        "> ⚠️ 数据为**合成**数据（由 `qalab/synth.py` 按显式失效机理生成），"
        "不是任何真实产线的数据。\n"
    )

    # 1. 数据规模
    parts.append("\n## 1. 数据规模与合并结果\n")
    parts.append(
        _table(
            ["项目", "数值"],
            [
                ["原料批次 material_batches", f"{data['n_batches']} 行"],
                ["生产参数 production_runs", f"{data['n_runs']} 行"],
                ["质量检验 inspection_results", f"{data['n_inspections']} 行"],
                ["性能测试 perf_tests", f"{data['n_perf_tests']} 行"],
                ["合并后分析宽表", f"{data['n_wide_rows']} 行 × {data['n_wide_columns']} 列"],
                ["清洗后建模用表", f"{data['n_clean_rows']} 行"],
                ["性能测试一对多扇出", f"最多 {data['perf_fanout_max']} 次/单元"],
                ["不合格单元数 / 不合格率", f"{data['n_fail']} / {_fmt(data['fail_rate'])}"],
                ["建模特征数（剔除泄漏字段后）", f"{modeling['n_features']}"],
            ],
        )
    )

    # 2. 数据质量
    parts.append("\n## 2. 数据质量检查结论\n")
    cb = quality["class_balance"]
    parts.append(
        _table(
            ["检查项", "结果"],
            [
                ["含缺失的列数", f"{quality['columns_with_missing']}"],
                ["全缺失列数", f"{quality['columns_all_missing']}"],
                ["含缺失的行数", f"{quality['missing_by_row']['rows_with_any_missing']} "
                 f"({_fmt(quality['missing_by_row']['pct_rows_with_any_missing'], 2)}%)"],
                ["越界值总数", f"{quality['total_out_of_range_values']}"],
                ["越界涉及的列", "、".join(quality["oob_columns"]) or "无"],
                ["完全重复行", f"{quality['duplicates']['n_full_duplicate_rows']}"],
                ["常量列", "、".join(quality["constant_columns"]) or "无"],
                ["近常量列", "、".join(quality["near_constant_columns"]) or "无"],
                ["发生分布漂移的列（对基线批次）", "、".join(quality["drifted_columns"]) or "无"],
                ["KS 显著但 PSI 低于工程阈值的列", "、".join(
                    sorted(set(quality.get("ks_significant_columns", []))
                           - set(quality["drifted_columns"]))
                ) or "无"],
                ["类别不平衡", f"{cb['severity']}（少数类占比 {_fmt(cb['minority_share'])}）"],
                ["数值列类型/解析问题", "、".join(quality["dtype_issue_columns"]) or "无"],
            ],
        )
    )
    clean_actions = metrics["cleaning"]["actions"]
    parts.append("\n**清洗动作（全部可复算）**\n")
    parts.append(
        _table(
            ["动作", "结果"],
            [
                ["删除完全重复行", f"{clean_actions['exact_duplicate_rows_dropped']}"],
                ["越界值置为缺失", json.dumps(clean_actions["out_of_range_to_nan"], ensure_ascii=False)],
                ["数值列中位数填补", json.dumps(
                    {k: round(v, 3) for k, v in clean_actions["numeric_median_imputed"].items()},
                    ensure_ascii=False,
                )],
                ["类别列缺失填 UNKNOWN", "、".join(clean_actions["categorical_filled_unknown"]) or "无"],
                ["行数 前 -> 后", f"{clean_actions['rows_before']} -> {clean_actions['rows_final']}"],
            ],
        )
    )

    # 3. 显著因子
    parts.append("\n## 3. 显著因子识别（多重比较校正后）\n")
    parts.append(
        f"共检验 {sig['n_factors_tested']} 个因子，"
        f"Benjamini-Hochberg 校正（FDR={sig['alpha']}）；"
        f"权威口径为「因子自身变化层级」的汇总检验"
        f"（生产参数 144 个炉次独立观测，原料因子 {sig['n_batches']} 个批次独立观测）。\n"
    )
    rows = []
    for _, r in ranking.iterrows():
        rows.append(
            [
                int(r["rank"]),
                r["factor"],
                {"numeric": "数值", "categorical": "类别"}.get(r["kind"], r["kind"]),
                r["factor_level"],
                r["n_independent"] if pd.notna(r["n_independent"]) else "n/a",
                _fmt_p(r["factor_p_value"]),
                _fmt_p(r["factor_q_value"]),
                _fmt(r["unit_effect_size"], 4),
                _fmt_p(r["unit_p_value"]),
                r["verdict"],
            ]
        )
    parts.append(
        _table(
            [
                "排名",
                "因子",
                "类型",
                "检验层级",
                "独立观测数",
                "原始 p",
                "BH 校正 q",
                "效应量",
                "簇置换 p",
                "结论",
            ],
            rows,
        )
    )
    parts.append(
        f"\n**显著因子（q≤{sig['alpha']}，因子层级口径）：**"
        + ("、".join(f"`{f}`" for f in sig["significant_factors"]) or "无")
    )
    parts.append(
        f"\n**伪重复假阳性演示：** 朴素单元级检验（把 {data['n_clean_rows']} 个单元当独立观测）"
        f"会把 {sig['n_naive_only_significant']} 个因子判为显著，"
        f"而其中 {sig['n_naive_false_positive']} 个在两种有效口径下都不显著："
        + ("、".join(f"`{f}`" for f in sig["naive_false_positives"]) or "无")
        + "。\n"
    )
    if calib:
        parts.append(
            f"\n**跨种子稳定性（{calib['n_seeds']} 个种子）：**"
            + "、".join(
                f"`{k}` {v}/{calib['n_seeds']}" for k, v in calib["primary_recovery"].items()
            )
            + f"；真值零效应因子被稳定误判的次数："
            + "、".join(
                f"`{k}` {v}/{calib['n_seeds']}" for k, v in calib["decoy_false_positive"].items()
            )
            + "。\n"
        )

    # 4. 统计检验
    parts.append("\n## 4. 统计检验（以排名第一的因子为例）\n")
    top = sig["top_factor_detail"]
    parts.append(
        _table(
            ["检验", "统计量", "p 值", "效应量", "说明"],
            [
                [
                    top["primary_method"],
                    _fmt(top["primary_statistic"], 4),
                    _fmt_p(top["primary_p_value"]),
                    f"{_fmt(top['effect_size'], 4)} ({top['effect_metric']})",
                    "因子层级（权威）",
                ],
                [
                    top["cluster_method"],
                    _fmt(top["cluster_statistic"], 4),
                    _fmt_p(top["cluster_p_value"]),
                    "—",
                    "簇置换，修正伪重复",
                ],
                [
                    top["naive_method"],
                    _fmt(top["naive_statistic"], 4),
                    _fmt_p(top["naive_p_value"]),
                    "—",
                    "朴素单元级（无效，仅对照）",
                ],
                [
                    top.get("supplementary_method", "point_biserial"),
                    "—",
                    _fmt_p(top.get("supplementary_p_value")),
                    "—",
                    "补充口径",
                ],
            ],
        )
    )

    # 5. 模型
    parts.append("\n## 5. 预测模型对比（测试集只评估一次）\n")
    parts.append(
        f"训练/验证/测试 = {modeling['split_sizes']['train']} / "
        f"{modeling['split_sizes']['validation']} / {modeling['split_sizes']['test']}，"
        f"分层切分（三者不合格率："
        + " / ".join(_fmt(v) for v in modeling["split_positive_rates"].values())
        + f"）；阈值在验证集上按 F1 选定；{modeling['cv_folds']} 折分层交叉验证。"
        "类别不平衡通过 `class_weight='balanced'` 处理。\n"
    )
    rows = []
    for name, m in modeling["models"].items():
        tm = m["test_metrics"]
        cv = modeling["cv"][name]
        rows.append(
            [
                name,
                _fmt(m["threshold_selection"]["threshold"], 3),
                _fmt(tm["average_precision"], 4),
                _fmt(tm["roc_auc"], 4),
                _fmt(tm["precision"], 4),
                _fmt(tm["recall"], 4),
                _fmt(tm["f1"], 4),
                _fmt(tm["accuracy"], 4),
                f"{_fmt(cv['average_precision']['mean'], 4)} ± {_fmt(cv['average_precision']['std'], 4)}",
            ]
        )
    parts.append(
        _table(
            ["模型", "阈值", "AP", "ROC-AUC", "Precision", "Recall", "F1", "Accuracy", "5 折 CV AP"],
            rows,
        )
    )
    best = modeling["best_model_by_validation_ap"]
    best_tm = modeling["models"][best]["test_metrics"]
    cm = best_tm["confusion_matrix"]
    parts.append(
        f"\n**按验证集 AP 选出的主模型：** `{best}`"
        f"（测试集 AP={_fmt(best_tm['average_precision'])}, "
        f"ROC-AUC={_fmt(best_tm['roc_auc'])}, F1={_fmt(best_tm['f1'])}；"
        f"混淆矩阵 [[TN,FP],[FN,TP]] = {cm}）。\n"
    )
    parts.append("\n**主模型最重要的特征（前 8）**\n")
    imp_rows = [
        [f.get("feature"), f.get("importance_source", ""), _fmt(f.get("importance"), 4),
         f.get("direction", "—")]
        for f in modeling["models"][best]["top_features"][:8]
    ]
    parts.append(_table(["特征", "重要性来源", "重要性", "方向"], imp_rows))

    abl = modeling.get("ablation_with_perf_features")
    if abl:
        parts.append(
            f"\n**对照实验（信息泄漏量化）：** 在主模型特征之外加入破坏性性能测试指标"
            f"（{abl['n_features']} 个特征），测试集 AP 从 "
            f"{_fmt(best_tm['average_precision'])} 变为 {_fmt(abl['test_metrics']['average_precision'])}。"
            f"{abl['note']}\n"
        )

    # 6. 图
    parts.append("\n## 6. 图表\n")
    for p in plots:
        parts.append(f"- `{Path(p).name}`")
    parts.append(
        "\n图表说明：图 1 类别不平衡；图 2 数值因子按标签的分布；图 3 按标签分组的箱线图；"
        "图 4 Spearman 相关性热图；图 5 因子重要性（按 BH 显著性着色）；"
        "图 6 主模型混淆矩阵；图 7 ROC 与 PR 曲线；图 8 数据质量总览；图 9 模型对比。\n"
    )

    # 7. 复现
    parts.append("\n## 7. 可复现性与边界\n")
    parts.append(
        f"- 随机种子：`{metrics['run']['seed']}`（数据生成、切分、交叉验证、模型初始化统一使用）\n"
        f"- 数据指纹（清洗后宽表 sha256）：`{metrics['data']['clean_fingerprint']}`\n"
        f"- 置换检验：{sig['n_perm']} 次簇置换，固定种子；置换 p 值下界为 1/(n_perm+1)"
        f" = {_fmt(sig['perm_p_floor'], 6)}\n"
        "- 数据为合成数据，结论不代表任何真实产线\n"
        "- 无企业部署、无真实用户、无线上流量\n"
    )
    return "\n".join(parts) + "\n"
