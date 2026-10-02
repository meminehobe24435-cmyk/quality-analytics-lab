"""可视化：以 seaborn 为主体的出图模块。

图里的文字统一用英文：CI（Ubuntu runner）上默认没有中文字体，
中文标签会变成一堆方块，属于典型的「本地好看、CI 出丑」的坑。

所有函数都保存 PNG 到 out_dir 并返回写出的路径列表，便于测试断言文件真的生成了。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 必须在 import pyplot 之前设置，否则无头环境下会报错

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

from . import config as cfg  # noqa: E402

__all__ = ["plot_all", "set_style"]

DPI = 140


def set_style() -> None:
    """统一绘图风格（seaborn 主题 + 中文字体无关的设置）。"""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "font.size": 10,
            "figure.autolayout": False,
        }
    )


# 图里的文字一律用英文：CI（Ubuntu runner）默认没有中文字体，
# 中文标签会变成一堆方块。timeseries 模块返回的趋势判定是中文，
# 直接塞进标题就会触发 "Glyph missing from font(s)" 并画出豆腐块 —— 实测踩过。
TREND_LABEL_EN = {
    "无显著趋势": "no significant trend",
    "显著上升趋势": "significant upward trend",
    "显著下降趋势": "significant downward trend",
    "无法判定": "undetermined",
}


def _trend_en(label: object) -> str:
    return TREND_LABEL_EN.get(str(label), str(label))


def _save(fig, out_dir: Path, name: str) -> Path:
    path = Path(out_dir) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# 单张图
# --------------------------------------------------------------------------
def plot_label_balance(df: pd.DataFrame, out_dir: Path, label: str = cfg.LABEL_COL) -> Path:
    """类别不平衡：合格/不合格的样本数与占比。"""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    counts = df[label].value_counts().sort_index()
    ax = axes[0]
    sns.barplot(
        x=["pass (0)", "fail (1)"][: len(counts)],
        y=counts.to_numpy(),
        hue=["pass (0)", "fail (1)"][: len(counts)],
        ax=ax,
        palette="deep",
        legend=False,
    )
    for i, v in enumerate(counts.to_numpy()):
        ax.text(i, v, f"{int(v)}", ha="center", va="bottom", fontsize=10)
    ax.set_title("Class balance: unit count")
    ax.set_ylabel("units")
    ax.set_xlabel("")

    ax = axes[1]
    rates = df.groupby("run_id")[label].mean()
    sns.histplot(rates, bins=20, ax=ax, color="darkorange")
    ax.set_title(f"Fail rate per run (mean={rates.mean():.3f})")
    ax.set_xlabel("fail rate within run")
    ax.set_ylabel("runs")
    fig.suptitle("Target overview — synthetic quality data", y=1.02)
    return _save(fig, out_dir, "01_label_balance.png")


def plot_numeric_distributions_by_label(
    df: pd.DataFrame, out_dir: Path, label: str = cfg.LABEL_COL, max_cols: int = 6
) -> Path:
    """数值因子按标签分组的分布（KDE）。"""
    cols = [c for c in cfg.MODEL_NUMERIC if c in df.columns][:max_cols]
    ncol = 3
    nrow = int(np.ceil(len(cols) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 2.8 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, col in zip(axes, cols):
        sns.kdeplot(
            data=df,
            x=col,
            hue=label,
            common_norm=False,
            fill=True,
            alpha=0.35,
            ax=ax,
        )
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
    for ax in axes[len(cols) :]:
        ax.axis("off")
    fig.suptitle("Numeric factor distributions by inspection outcome (0=pass, 1=fail)", y=1.01)
    return _save(fig, out_dir, "02_numeric_distributions_by_label.png")


def plot_boxplots_by_label(
    df: pd.DataFrame, out_dir: Path, label: str = cfg.LABEL_COL
) -> Path:
    """数值因子按标签分组的箱线图（含每组的离散程度对比）。"""
    cols = [c for c in cfg.MODEL_NUMERIC if c in df.columns]
    melted = df[[label, *cols]].melt(id_vars=label, var_name="factor", value_name="value")
    # 各列量纲差异极大，按列做 z-score 标准化后同图比较
    melted["value_z"] = melted.groupby("factor")["value"].transform(
        lambda s: (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) else 1.0)
    )
    fig, ax = plt.subplots(figsize=(10, 4.6))
    sns.boxplot(
        data=melted,
        x="factor",
        y="value_z",
        hue=label,
        ax=ax,
        palette={0: "#4C72B0", 1: "#C44E52"},
        fliersize=2,
    )
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_title("Numeric factors by label (z-scored per factor; 0=pass, 1=fail)")
    ax.set_xlabel("")
    ax.set_ylabel("z-score within factor")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    ax.legend(title="is_fail", ncol=2)
    return _save(fig, out_dir, "03_boxplots_by_label.png")


def plot_correlation_heatmap(
    df: pd.DataFrame, out_dir: Path, label: str = cfg.LABEL_COL
) -> Path:
    """相关性热图（Spearman，对非线性单调关系与离群值更稳健）。"""
    cols = [c for c in cfg.ALL_NUMERIC if c in df.columns]
    if label in df.columns:
        cols = [*cols, label]
    corr = df[cols].corr(method="spearman", numeric_only=True)
    fig, ax = plt.subplots(figsize=(8.4, 6.8))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Spearman rho"},
        annot_kws={"size": 7},
        ax=ax,
    )
    ax.set_title(f"Spearman correlation among numeric factors and target ({label})")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    return _save(fig, out_dir, "04_correlation_heatmap.png")


def plot_factor_importance(
    ranking: pd.DataFrame, out_dir: Path, top_k: int = 12
) -> Path:
    """因子重要性条形图：效应量大小 + 是否显著（着色）。"""
    top = ranking.head(top_k).copy()
    top["abs_effect"] = top["unit_effect_size"].abs()
    top = top.sort_values("abs_effect", ascending=True)
    top["verdict_short"] = np.where(
        top["factor_significant"], "significant (BH q<=0.05)", "not significant"
    )
    fig, ax = plt.subplots(figsize=(9.2, 0.5 * len(top) + 2.0))
    sns.barplot(
        data=top,
        y="factor",
        x="abs_effect",
        hue="verdict_short",
        dodge=False,
        palette={"significant (BH q<=0.05)": "#C44E52", "not significant": "#B0B0B0"},
        ax=ax,
    )
    for i, (_, row) in enumerate(top.iterrows()):
        ax.text(
            row["abs_effect"],
            i,
            f"  q={row['factor_q_value']:.2e}" if pd.notna(row["factor_q_value"]) else "  q=NA",
            va="center",
            fontsize=8,
        )
    ax.set_title("Factor importance (|effect size| at unit level), colored by BH significance")
    ax.set_xlabel("|effect size| (rank-biserial for numeric, Cramer's V for categorical)")
    ax.set_ylabel("")
    ax.legend(title="verdict", loc="lower right")
    ax.set_xlim(0, float(top["abs_effect"].max()) * 1.35 if len(top) else 1.0)
    return _save(fig, out_dir, "05_factor_importance.png")


def plot_confusion_matrix(
    cm: list[list[int]], out_dir: Path, model_name: str
) -> Path:
    """混淆矩阵热图（测试集，阈值在验证集上选定）。"""
    arr = np.asarray(cm, dtype=float)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    sns.heatmap(
        arr,
        annot=np.array([[f"{int(v)}" for v in row] for row in arr]),
        fmt="",
        cmap="Blues",
        cbar=False,
        square=True,
        linewidths=1,
        linecolor="white",
        ax=ax,
    )
    ax.set_xticklabels(["pred pass", "pred fail"])
    ax.set_yticklabels(["actual pass", "actual fail"], rotation=0)
    ax.set_title(f"Confusion matrix — {model_name}\n(test set, threshold from validation)")
    ax.set_xlabel("")
    ax.set_ylabel("")
    return _save(fig, out_dir, "06_confusion_matrix.png")


def plot_roc_pr_curves(
    y_test, test_proba: dict[str, np.ndarray], out_dir: Path, ap_scores: dict[str, float]
) -> Path:
    """ROC 与 PR 曲线并排；PR 曲线在不平衡数据上比 ROC 更能反映实际收益。"""
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))
    for name, proba in test_proba.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={ap_scores[name]['roc_auc']:.3f})")
        prec, rec, _ = precision_recall_curve(y_test, proba)
        axes[1].plot(rec, prec, label=f"{name} (AP={ap_scores[name]['average_precision']:.3f})")
    axes[0].plot([0, 1], [0, 1], ls="--", color="grey", lw=0.8)
    axes[0].set_xlabel("false positive rate")
    axes[0].set_ylabel("true positive rate")
    axes[0].set_title("ROC curve (test set)")
    axes[0].legend(fontsize=8, loc="lower right")

    base = float(np.mean(y_test))
    axes[1].axhline(base, ls="--", color="grey", lw=0.8, label=f"baseline={base:.3f}")
    axes[1].set_xlabel("recall")
    axes[1].set_ylabel("precision")
    axes[1].set_title("Precision-Recall curve (test set)")
    axes[1].legend(fontsize=8, loc="lower left")
    return _save(fig, out_dir, "07_roc_pr_curves.png")


def plot_data_quality(
    quality_summary: dict[str, object],
    missing_table: pd.DataFrame,
    oob_table: pd.DataFrame,
    out_dir: Path,
) -> Path:
    """数据质量总览：缺失率 + 越界计数（两张子图）。"""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

    miss = missing_table[missing_table["n_missing"] > 0].copy()
    ax = axes[0]
    if len(miss):
        miss = miss.sort_values("pct_missing", ascending=False).head(15)
        sns.barplot(data=miss, y="column", x="pct_missing", ax=ax, color="#4C72B0")
        ax.set_xlabel("% missing")
        ax.set_title("Columns with missing values")
    else:
        ax.text(0.5, 0.5, "no missing values", ha="center", va="center")
        ax.set_axis_off()
    ax.set_ylabel("")

    ax = axes[1]
    oob = oob_table[oob_table["n_out_of_range"] > 0].copy()
    if len(oob):
        long = oob.melt(
            id_vars="column",
            value_vars=["n_below", "n_above"],
            var_name="kind",
            value_name="n",
        )
        sns.barplot(data=long, y="column", x="n", hue="kind", ax=ax)
        ax.set_xlabel("count of out-of-range values")
        ax.set_title("Out-of-spec-limit counts")
    else:
        ax.text(0.5, 0.5, "no out-of-range values", ha="center", va="center")
        ax.set_axis_off()
    ax.set_ylabel("")

    const = quality_summary.get("constant_columns") or []
    near = quality_summary.get("near_constant_columns") or []
    fig.suptitle(
        f"Data quality overview — {quality_summary.get('n_rows')} rows; "
        f"constant columns: {len(const)}; near-constant: {len(near)}",
        y=1.02,
    )
    return _save(fig, out_dir, "08_data_quality.png")


def plot_model_comparison(result: dict[str, object], out_dir: Path) -> Path:
    """模型对比：测试集 AP / ROC-AUC / F1（含神经网络基线）。"""
    models = result["models"]
    names = list(models)
    rows = []
    for name in names:
        tm = models[name]["test_metrics"]
        cv = result["cv"][name]["average_precision"]
        rows.append(
            {
                "model": name,
                "metric": "AP",
                "value": tm["average_precision"],
                "cv_mean": cv["mean"],
                "cv_std": cv["std"],
            }
        )
        rows.append(
            {
                "model": name,
                "metric": "ROC-AUC",
                "value": tm["roc_auc"],
                "cv_mean": result["cv"][name]["roc_auc"]["mean"],
                "cv_std": result["cv"][name]["roc_auc"]["std"],
            }
        )
        rows.append(
            {
                "model": name,
                "metric": "F1",
                "value": tm["f1"],
                "cv_mean": None,
                "cv_std": None,
            }
        )
    long = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    sns.barplot(data=long, x="model", y="value", hue="metric", ax=ax)
    ax.set_ylim(0, 1.0)
    ax.set_title("Model comparison on the held-out test set (evaluated once)")
    ax.set_ylabel("score")
    ax.set_xlabel("")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=7, padding=2)
    ax.legend(title="metric", ncol=3, loc="lower right")
    plt.setp(ax.get_xticklabels(), rotation=12, ha="right")
    return _save(fig, out_dir, "09_model_comparison.png")


# --------------------------------------------------------------------------
# 聚类 / 回归 / 时间序列
# --------------------------------------------------------------------------
def plot_cluster_selection(silhouette_table: pd.DataFrame, out_dir: Path) -> Path:
    """k 的选择：轮廓系数（越大越好）与 inertia（肘部）双面板。"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    table = silhouette_table.sort_values("k")
    ax = axes[0]
    sns.lineplot(data=table, x="k", y="silhouette", marker="o", ax=ax, color="#4C72B0")
    best = table.loc[table["silhouette"].idxmax()]
    ax.scatter([best["k"]], [best["silhouette"]], s=110, facecolor="none", edgecolor="#C44E52", zorder=5)
    ax.annotate(
        f"selected k={int(best['k'])}\nsilhouette={best['silhouette']:.3f}",
        xy=(best["k"], best["silhouette"]),
        xytext=(6, -18),
        textcoords="offset points",
        fontsize=8,
        color="#C44E52",
    )
    ax.set_title("Silhouette score by k (higher is better)")
    ax.set_xlabel("k (number of failure-mode clusters)")
    ax.set_ylabel("silhouette")
    if len(table) > 1:
        ax.set_xticks(table["k"].astype(int))

    ax = axes[1]
    sns.lineplot(data=table, x="k", y="inertia", marker="s", ax=ax, color="darkorange")
    ax.set_title("Within-cluster sum of squares (elbow)")
    ax.set_xlabel("k")
    ax.set_ylabel("inertia")
    if len(table) > 1:
        ax.set_xticks(table["k"].astype(int))
    fig.suptitle("Failure-mode clustering: choosing k", y=1.02)
    return _save(fig, out_dir, "10_cluster_selection.png")


def plot_cluster_profiles(profile: pd.DataFrame, feature_cols: list[str], out_dir: Path) -> Path:
    """簇画像热图：每个簇在每个特征上相对总体的偏离（z 值）。"""
    z_cols = [f"z_{c}" for c in feature_cols if f"z_{c}" in profile.columns]
    mat = profile.set_index("cluster")[z_cols].copy()
    mat.columns = [c[2:] for c in mat.columns]
    fig, ax = plt.subplots(figsize=(1.15 * len(mat.columns) + 3.2, 0.62 * len(mat) + 2.2))
    sns.heatmap(
        mat,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=0,
        linewidths=0.5,
        cbar_kws={"label": "deviation from overall mean (z)", "shrink": 0.8},
        annot_kws={"size": 8},
        ax=ax,
    )
    sizes = profile.set_index("cluster")["n"].to_dict()
    ax.set_yticklabels(
        [f"cluster {int(c)} (n={sizes.get(c, '?')})" for c in mat.index], rotation=0
    )
    ax.set_title("Failure-mode profiles: deviation from overall mean (z-scored)")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    return _save(fig, out_dir, "11_cluster_profiles.png")


def plot_cluster_defect_composition(composition: pd.DataFrame, out_dir: Path) -> Path:
    """每个簇的缺陷类型构成（堆叠柱）—— 簇画像与业务语言的接口。"""
    if composition.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "no defect composition available", ha="center", va="center")
        ax.set_axis_off()
        return _save(fig, out_dir, "12_cluster_defect_composition.png")

    long = (
        composition.reset_index()
        .melt(id_vars="cluster", var_name="defect_type", value_name="share")
        .sort_values(["cluster", "defect_type"])
    )
    long["cluster_label"] = "cluster " + long["cluster"].astype(int).astype(str)
    fig, ax = plt.subplots(figsize=(9.4, 4.4))
    sns.barplot(
        data=long,
        x="cluster_label",
        y="share",
        hue="defect_type",
        ax=ax,
        palette="deep",
    )
    ax.set_title("Defect-type composition within each failure-mode cluster")
    ax.set_xlabel("")
    ax.set_ylabel("share of units in cluster")
    ax.set_ylim(0, 1.0)
    ax.legend(title="defect type", ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    return _save(fig, out_dir, "12_cluster_defect_composition.png")


def plot_cluster_scatter(pca: dict, labels, out_dir: Path) -> Path:
    """把高维簇用 PCA 投影到二维看一眼（仅用于可视化，不参与任何统计结论）。"""
    coords = np.asarray(pca["coords"])
    evr = pca.get("explained_variance_ratio", [0, 0])
    df = pd.DataFrame(
        {"pc1": coords[:, 0], "pc2": coords[:, 1], "cluster": [f"cluster {c}" for c in labels]}
    )
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    sns.scatterplot(data=df, x="pc1", y="pc2", hue="cluster", s=24, alpha=0.75, ax=ax)
    ax.set_title(
        f"Fail samples in PCA space (PC1 {evr[0]*100:.1f}% / PC2 {evr[1]*100:.1f}% variance)"
    )
    ax.legend(title="", fontsize=8, ncol=2)
    return _save(fig, out_dir, "13_cluster_pca_scatter.png")


def plot_regression_predicted_vs_actual(
    y_test: np.ndarray,
    predictions: dict[str, np.ndarray],
    baseline: np.ndarray,
    target: str,
    metrics: dict[str, dict],
    out_dir: Path,
) -> Path:
    """回归：预测值 vs 真值散点（含均值基线的水平线），每个模型一个面板。"""
    names = list(predictions)
    ncol = min(2, max(1, len(names)))
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 4.4 * nrow), squeeze=False)
    axes = axes.ravel()
    lo = float(min(y_test.min(), *(p.min() for p in predictions.values())))
    hi = float(max(y_test.max(), *(p.max() for p in predictions.values())))
    for ax, name in zip(axes, names):
        pred = predictions[name]
        sns.scatterplot(x=y_test, y=pred, s=14, alpha=0.45, ax=ax, color="#4C72B0", edgecolor="none")
        ax.plot([lo, hi], [lo, hi], ls="--", lw=1, color="black", label="perfect prediction")
        ax.axhline(float(np.mean(baseline)), ls=":", lw=1.2, color="#C44E52", label="mean baseline")
        m = metrics.get(name, {})
        ax.set_title(
            f"{name}\nMAE={m.get('mae', float('nan')):.3f}  RMSE={m.get('rmse', float('nan')):.3f}  "
            f"R2={m.get('r2', float('nan')):.3f}"
        )
        ax.set_xlabel(f"actual {target}")
        ax.set_ylabel("predicted")
        ax.legend(fontsize=7, loc="upper left")
    for ax in axes[len(names) :]:
        ax.axis("off")
    fig.suptitle("Regression on a continuous quality metric (test set)", y=1.02)
    return _save(fig, out_dir, "14_regression_predicted_vs_actual.png")


def plot_timeseries_trend(
    series: dict[str, object], backtest: dict[str, object], series_name: str, out_dir: Path
) -> Path:
    """时间序列：批次序列 + 移动平均 + 线性趋势（左），一步预测回测误差（右）。"""
    values = pd.Series([np.nan if v is None else v for v in series.get("values", [])], dtype="float64")
    ma = pd.Series(
        [np.nan if v is None else v for v in series.get("moving_average", [])], dtype="float64"
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.2))

    ax = axes[0]
    x = np.arange(len(values))
    ax.plot(x, values, marker="o", ms=3.5, lw=1.2, color="#4C72B0", label="observed")
    if ma.notna().any():
        ax.plot(x, ma, lw=2.0, color="darkorange", label=f"moving average (w={series.get('window')})")
    lt = series.get("linear_trend", {})
    if lt.get("slope") is not None:
        intercept = lt["intercept"]
        ax.plot(
            x,
            intercept + lt["slope"] * x,
            ls="--",
            lw=1.4,
            color="#C44E52",
            label=f"linear trend (p={lt['p_value']:.3f})",
        )
    mk = series.get("mann_kendall", {})
    ax.set_title(
        f"{series_name}\nMann-Kendall: {_trend_en(mk.get('trend'))} "
        f"(p={mk.get('p_value') if mk.get('p_value') is None else round(mk['p_value'], 3)})"
    )
    ax.set_xlabel("batch order (production sequence)")
    ax.set_ylabel(series_name.replace("_", " "))
    ax.legend(fontsize=8)

    ax = axes[1]
    predictors = backtest.get("predictors", {}) if backtest else {}
    if predictors:
        frame = pd.DataFrame(
            [{"predictor": k, "MAE": v["mae"], "RMSE": v["rmse"]} for k, v in predictors.items()]
        ).melt(id_vars="predictor", var_name="metric", value_name="error")
        sns.barplot(data=frame, x="predictor", y="error", hue="metric", ax=ax)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2)
        ax.set_title("One-step-ahead forecast error (expanding-window backtest)")
        ax.set_ylabel("error")
        ax.set_xlabel("")
        ax.legend(title="", fontsize=8)
    else:
        ax.text(0.5, 0.5, "backtest unavailable", ha="center", va="center")
        ax.set_axis_off()
    fig.suptitle("Batch-ordered series analysis (not high-frequency time series)", y=1.02)
    path = _save(fig, out_dir, "15_timeseries_trend.png")
    return path



# --------------------------------------------------------------------------
# 统一入口
# --------------------------------------------------------------------------
def plot_all(
    clean_df: pd.DataFrame,
    ranking: pd.DataFrame,
    quality_summary: dict[str, object],
    missing_table: pd.DataFrame,
    oob_table: pd.DataFrame,
    modeling_result: dict[str, object],
    modeling_extras: dict[str, object],
    out_dir: Path,
    extra_analyses: dict[str, object] | None = None,
) -> list[Path]:
    """生成全部图，返回写出的文件路径列表。

    `extra_analyses` 传入 {"clustering": (result, extras), "regression": (result, extras),
    "timeseries": (result, extras)} 时，额外产出聚类/回归/时序的 5 张图；
    不传则只产出前 9 张（保持向后兼容）。
    """
    set_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    paths.append(plot_label_balance(clean_df, out_dir))
    paths.append(plot_numeric_distributions_by_label(clean_df, out_dir))
    paths.append(plot_boxplots_by_label(clean_df, out_dir))
    paths.append(plot_correlation_heatmap(clean_df, out_dir))
    if not ranking.empty:
        paths.append(plot_factor_importance(ranking, out_dir))
    paths.append(plot_data_quality(quality_summary, missing_table, oob_table, out_dir))
    paths.append(plot_model_comparison(modeling_result, out_dir))

    y_test = modeling_extras["y_test"]
    ap_scores = {
        name: modeling_result["models"][name]["test_metrics"]
        for name in modeling_result["models"]
    }
    paths.append(
        plot_roc_pr_curves(y_test, modeling_extras["test_proba"], out_dir, ap_scores)
    )
    best = modeling_extras["best_model"]
    paths.append(plot_confusion_matrix(modeling_extras["confusion_matrices"][best], out_dir, best))

    if extra_analyses:
        cluster_part = extra_analyses.get("clustering")
        if cluster_part:
            c_result, c_extras = cluster_part
            sil = c_extras.get("silhouette_table")
            if sil is not None and len(sil):
                paths.append(plot_cluster_selection(sil, out_dir))
            profile = c_extras.get("profile")
            if profile is not None and len(profile):
                feature_cols = [c[2:] for c in profile.columns if c.startswith("z_")]
                paths.append(plot_cluster_profiles(profile, feature_cols, out_dir))
            composition = c_extras.get("composition")
            if composition is not None and len(composition):
                paths.append(plot_cluster_defect_composition(composition, out_dir))
            pca = c_extras.get("pca")
            if pca and c_extras.get("labels") is not None:
                paths.append(plot_cluster_scatter(pca, c_extras["labels"], out_dir))

        reg_part = extra_analyses.get("regression")
        if reg_part:
            r_result, r_extras = reg_part
            if r_extras.get("pred_test"):
                paths.append(
                    plot_regression_predicted_vs_actual(
                        r_extras["y_test"],
                        r_extras["pred_test"],
                        r_extras["baseline_test"],
                        r_result.get("primary_target", "target"),
                        {
                            n: r_result["primary"]["models"][n]["test"]
                            for n in r_result["primary"].get("models", {})
                        },
                        out_dir,
                    )
                )

        ts_part = extra_analyses.get("timeseries")
        if ts_part:
            ts_result, _ts_extras = ts_part
            primary = ts_result.get("series", {}).get("batch_fail_rate")
            if primary and primary.get("n_points", 0) >= 2:
                paths.append(
                    plot_timeseries_trend(
                        primary,
                        primary.get("backtest", {}),
                        "batch fail rate",
                        out_dir,
                    )
                )
    return paths

