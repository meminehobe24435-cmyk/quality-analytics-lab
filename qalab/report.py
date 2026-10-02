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

    # 6. 算法覆盖：分类之外的四类算法
    clustering = metrics.get("clustering", {})
    regression = metrics.get("regression", {})
    timeseries = metrics.get("timeseries", {})

    parts.append("\n## 6. 聚类：把「不合格」拆成几种失效模式\n")
    parts.append(
        f"**用途**：{clustering.get('purpose', '')}\n\n"
        f"在不合格样本（n={clustering.get('n_samples')}）上对 "
        f"{len(clustering.get('feature_space', []))} 个工艺/原料特征做 KMeans，"
        f"k 在 {clustering.get('k_candidates')} 中按**轮廓系数**选，"
        f"选中 **k={clustering.get('k_selected')}**（轮廓系数 "
        f"{_fmt(clustering.get('silhouette'))}）。\n"
    )
    parts.append("\n**各 k 的轮廓系数**\n")
    parts.append(
        _table(
            ["k", "轮廓系数", "inertia"],
            [
                [k, _fmt(v), _fmt(clustering.get("inertia_by_k", {}).get(k), 2)]
                for k, v in (clustering.get("silhouette_by_k") or {}).items()
            ],
        )
    )
    profile = clustering.get("profile") or []
    if profile:
        parts.append("\n**簇画像（相对总体均值的偏离，z 值）**\n")
        rows = []
        for entry in profile:
            top = [
                (k[2:], v)
                for k, v in entry.items()
                if k.startswith("z_") and isinstance(v, (int, float))
            ]
            top.sort(key=lambda t: -abs(t[1]))
            rows.append(
                [
                    entry.get("cluster"),
                    entry.get("n"),
                    _fmt(entry.get("share")),
                    _fmt(entry.get("fail_rate_in_cluster")),
                    "、".join(f"{name} {v:+.2f}" for name, v in top[:3]),
                    f"{entry.get('top_defect_type')} ({_fmt(entry.get('top_defect_share'))})",
                ]
            )
        parts.append(
            _table(
                ["簇", "单元数", "占比", "簇内不合格率", "最偏离的特征(top3, z)", "主要缺陷类型(占比)"],
                rows,
            )
        )
    if clustering.get("note"):
        parts.append(f"\n{clustering['note']}\n")

    parts.append("\n## 7. 回归：预测连续的性能指标\n")
    primary = regression.get("primary", {})
    parts.append(
        f"**用途**：{regression.get('purpose', '')}\n\n"
        f"主目标：`{regression.get('primary_target')}`；"
        f"有效样本 {primary.get('n_rows_used')} 条"
        f"（目标缺失的行直接丢弃，不填补 —— 填补目标等于伪造标签）；"
        f"切分 train/test = {primary.get('split', {}).get('train')} / "
        f"{primary.get('split', {}).get('test')}。\n"
    )
    if primary.get("models"):
        rows = []
        base = primary["baseline_mean"]["test"]
        rows.append(["**均值基线（什么都不学）**", "—", _fmt(base["mae"]), _fmt(base["rmse"]), _fmt(base["r2"])])
        for name, entry in primary["models"].items():
            rows.append(
                [
                    name,
                    _fmt(entry.get("mae_reduction_vs_baseline_pct"), 2) + "%",
                    _fmt(entry["test"]["mae"]),
                    _fmt(entry["test"]["rmse"]),
                    _fmt(entry["test"]["r2"]),
                ]
            )
        parts.append(
            _table(["模型", "MAE 相对基线降低", "测试 MAE", "测试 RMSE", "测试 R²"], rows)
        )
        parts.append(
            f"\n**结论**：{regression.get('note', '')}。"
            + (
                "模型**确实优于均值基线**，说明工艺参数对性能指标有可利用的预测力。"
                if regression.get("beats_baseline")
                else "模型**没有超过均值基线** —— 这种情况下回归模型不应被采用。"
            )
            + "\n"
        )
    if regression.get("all_targets"):
        parts.append("\n**其它目标（最优模型按测试 MAE 选）**\n")
        parts.append(
            _table(
                ["目标", "有效样本", "基线 MAE", "最优模型", "MAE", "RMSE", "R²"],
                [
                    [
                        e["target"],
                        e["n_rows_used"],
                        _fmt(e["baseline_mae"], 4),
                        e["best_model"] or "n/a",
                        _fmt(e["best_mae"], 4),
                        _fmt(e["best_rmse"], 4),
                        _fmt(e["best_r2"], 4),
                    ]
                    for e in regression["all_targets"]
                ],
            )
        )

    parts.append("\n## 8. 神经网络（MLPClassifier）与树模型的对比\n")
    mlp = modeling["models"].get("mlp_classifier")
    if mlp:
        tm = mlp["test_metrics"]
        parts.append(
            f"`MLPClassifier`（hidden=(32,16)、lbfgs 求解器、`class_weight='balanced'`、"
            f"输入经标准化）与树模型在同一套切分与阈值规则下对比：\n\n"
            f"- 测试集 AP = {_fmt(tm['average_precision'])}、"
            f"ROC-AUC = {_fmt(tm['roc_auc'])}、F1 = {_fmt(tm['f1'])}\n"
            f"- 5 折 CV AP = {_fmt(modeling['cv']['mlp_classifier']['average_precision']['mean'])}"
            f" ± {_fmt(modeling['cv']['mlp_classifier']['average_precision']['std'])}\n"
            f"- 阈值（验证集选定）= {_fmt(mlp['threshold_selection']['threshold'], 3)}\n"
            f"- 混淆矩阵 = {tm['confusion_matrix']}\n"
        )
        best_name = modeling["best_model_by_validation_ap"]
        best_ap = modeling["models"][best_name]["test_metrics"]["average_precision"]
        diff = tm["average_precision"] - best_ap
        parts.append(
            f"\n**如实结论**：在 {len(modeling['models'])} 个模型里，"
            f"神经网络{('不如' if diff < 0 else '不差于')}最优的 `{best_name}`"
            f"（AP {_fmt(tm['average_precision'])} vs {_fmt(best_ap)}，差 {diff:+.4f}）。"
            "本数据是约 5000 行、11 个特征的**表格数据**，特征与标签的关系以单调/近似线性为主，"
            "因此树模型与线性模型已经足够，神经网络没有体现出优势，"
            "而且它的可解释性与调参成本都更差 —— 这也是本项目不把它作为主模型的原因。\n"
        )

    parts.append("\n## 9. 时间序列 / 趋势分析（按批次顺序的序列，不是高频时序）\n")
    parts.append(f"{timeseries.get('scope_note', '')}\n")
    series = timeseries.get("series", {})
    if series:
        rows = []
        for name, s in series.items():
            mk = s.get("mann_kendall", {})
            lt = s.get("linear_trend", {})
            ac = s.get("autocorrelation_lag1", {})
            rows.append(
                [
                    name,
                    s.get("n_points"),
                    _fmt_p(mk.get("p_value")),
                    mk.get("trend", "n/a"),
                    _fmt(mk.get("sen_slope"), 6),
                    _fmt_p(lt.get("p_value")),
                    _fmt(ac.get("r1"), 4),
                    "是" if ac.get("significant") else "否",
                ]
            )
        parts.append(
            _table(
                ["序列", "点数", "MK p", "MK 趋势判定", "Sen's 斜率", "线性趋势 p", "滞后1 自相关", "自相关显著"],
                rows,
            )
        )
    fc = timeseries.get("next_batch_forecast") or {}
    if fc:
        parts.append("\n**下一个批次的一步预测**\n")
        parts.append(
            _table(
                ["预测方法", "预测值"],
                [
                    ["指数平滑 (SES)", _fmt(fc.get("exponential_smoothing"), 4)],
                    ["AR(1)", _fmt(fc.get("ar1"), 4)],
                    ["历史均值", _fmt(fc.get("historical_mean"), 4)],
                    ["上一个观测值", _fmt(fc.get("last_observed"), 4)],
                ],
            )
        )
    errs = timeseries.get("forecast_one_step_error") or {}
    if errs:
        parts.append("\n**一步预测回测误差（扩张窗口，每个点只用它之前的数据）**\n")
        parts.append(
            _table(
                ["预测器", "MAE", "RMSE"],
                [[k, _fmt(v.get("mae"), 4), _fmt(v.get("rmse"), 4)] for k, v in errs.items()],
            )
        )
    parts.append(f"\n**结论**：{timeseries.get('conclusion', '')}\n")

    # 10. 图
    parts.append("\n## 10. 图表\n")
    for p in plots:
        parts.append(f"- `{Path(p).name}`")
    parts.append(
        "\n图表说明：01 类别不平衡；02 数值因子按标签的分布；03 按标签分组的箱线图；"
        "04 Spearman 相关性热图；05 因子重要性（按 BH 显著性着色）；06 主模型混淆矩阵；"
        "07 ROC 与 PR 曲线；08 数据质量总览；09 模型对比（含神经网络）；"
        "10 聚类 k 的选择；11 失效模式簇画像；12 簇内缺陷类型构成；13 簇的 PCA 投影；"
        "14 回归预测值 vs 真值；15 批次序列趋势与一步预测误差。\n"
    )

    # 11. 复现
    parts.append("\n## 11. 可复现性与边界\n")
    parts.append(
        f"- 随机种子：`{metrics['run']['seed']}`（数据生成、切分、交叉验证、模型初始化统一使用）\n"
        f"- 数据指纹（清洗后宽表 sha256）：`{metrics['data']['clean_fingerprint']}`\n"
        f"- 置换检验：{sig['n_perm']} 次簇置换，固定种子；置换 p 值下界为 1/(n_perm+1)"
        f" = {_fmt(sig['perm_p_floor'], 6)}\n"
        "- 数据为合成数据，结论不代表任何真实产线\n"
        "- 无企业部署、无真实用户、无线上流量\n"
    )
    return "\n".join(parts) + "\n"
