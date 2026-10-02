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
    """模型对比：测试集 AP / ROC-AUC / F1 + 交叉验证 AP（带误差棒）。"""
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
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    sns.barplot(data=long, x="model", y="value", hue="metric", ax=ax)
    ax.set_ylim(0, 1.0)
    ax.set_title("Model comparison on the held-out test set (evaluated once)")
    ax.set_ylabel("score")
    ax.set_xlabel("")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2)
    ax.legend(title="metric", ncol=3, loc="lower right")
    plt.setp(ax.get_xticklabels(), rotation=12, ha="right")
    return _save(fig, out_dir, "09_model_comparison.png")


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
) -> list[Path]:
    """生成全部图，返回写出的文件路径列表。"""
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
    return paths
