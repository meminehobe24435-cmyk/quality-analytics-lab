"""聚类分析：把「不合格」拆成几种可分别处置的失效模式。

**用途**：不合格标签是二值的，但"不合格"本身是一团混合物 —— 有的来自炉温过热、
有的来自原料含水率偏高、有的来自问题机台。把它们按工艺参数的相似性分成若干簇，
就能针对每一类失效模式分别给出处置方向（调温、换料、修机台），
而不是所有不合格都套同一条纠偏措施。

方法：
  1. 只对**不合格样品**做聚类（合格样品不在这个问题的范围内）；
  2. 特征用「检验之前就能拿到的」工艺参数与原料参数（与分类模型同一套特征，
     因此簇的画像可以直接翻译成"这一炉的工艺条件长什么样"）；
  3. k 在候选集合里用**轮廓系数**选（同时报告 inertia / Calinski-Harabasz /
     Davies-Bouldin 作为旁证，避免只看一个指标）；
  4. 簇画像 = 每个特征相对总体均值的偏离（以总体标准差为单位）+ 簇规模 +
     簇内主要缺陷类型占比。

退化输入（全部有测试）：不合格样本太少、k 候选不可行、特征方差为 0。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config as cfg

__all__ = [
    "prepare_cluster_matrix",
    "select_k",
    "fit_clusters",
    "cluster_profile",
    "defect_composition",
    "run_clustering",
]

K_CANDIDATES: tuple[int, ...] = (2, 3, 4, 5, 6)
"""k 的候选集合（对应故障诊断可解释的簇数上限，再多人就没法分别处置了）。"""


def prepare_cluster_matrix(
    df: pd.DataFrame, feature_cols: tuple[str, ...] | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """取出聚类用的数值特征矩阵（缺失用中位数填补，标准化在管线里做）。

    只用数值型工艺/原料参数：KMeans 的欧氏距离对 one-hot 类别列的权重没有客观依据，
    因此类别信息（机台、班次）不进特征空间，而是在簇画像阶段做事后刻画。
    """
    cols = [c for c in (feature_cols or cfg.MODEL_NUMERIC) if c in df.columns]
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    return X, cols


def select_k(
    X: pd.DataFrame, k_candidates: tuple[int, ...] = K_CANDIDATES, seed: int = cfg.SEED
) -> pd.DataFrame:
    """在每个候选 k 上拟合 KMeans，报告轮廓系数与辅助指标。

    k 必须落在 [2, n_samples - 1]（轮廓系数在 k=1 或 k=n 时无定义），
    因此样本量不足时候选集合会被自动收缩；一个候选都不剩时返回空表并说明原因。
    """
    n = len(X)
    rows: list[dict[str, object]] = []
    usable = [k for k in k_candidates if 2 <= k <= n - 1]
    for k in usable:
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("kmeans", KMeans(n_clusters=k, n_init=10, random_state=seed)),
            ]
        )
        labels = model.fit_predict(X)
        Xs = model.named_steps["scaler"].transform(
            model.named_steps["imputer"].transform(X)
        )
        rows.append(
            {
                "k": int(k),
                "silhouette": float(silhouette_score(Xs, labels)),
                "inertia": float(model.named_steps["kmeans"].inertia_),
                "calinski_harabasz": float(calinski_harabasz_score(Xs, labels)),
                "davies_bouldin": float(davies_bouldin_score(Xs, labels)),
            }
        )
    out = pd.DataFrame(rows, columns=["k", "silhouette", "inertia", "calinski_harabasz", "davies_bouldin"])
    if not out.empty:
        out = out.sort_values("k").reset_index(drop=True)
    return out


def fit_clusters(
    X: pd.DataFrame, k: int, seed: int = cfg.SEED
) -> tuple[np.ndarray, Pipeline]:
    """在给定 k 上拟合，返回 (簇标签, 已拟合的管线)。"""
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("kmeans", KMeans(n_clusters=k, n_init=10, random_state=seed)),
        ]
    )
    labels = model.fit_predict(X)
    return np.asarray(labels), model


def cluster_profile(
    df: pd.DataFrame,
    labels: np.ndarray,
    feature_cols: list[str],
    label: str = cfg.LABEL_COL,
) -> pd.DataFrame:
    """簇画像：规模 + 每个特征相对总体均值的偏离（以总体标准差为单位）。

    偏离用 z 值表示，正号=该簇在这个特征上比总体更高。z 值比原始均值更可读：
    不同特征量纲差异极大（炉温 1500℃ 与压力 3MPa 没法直接比）。
    """
    work = df.loc[:, feature_cols].apply(pd.to_numeric, errors="coerce").copy()
    work["_cluster"] = labels
    overall_mean = work[feature_cols].mean()
    overall_std = work[feature_cols].std(ddof=0).replace(0.0, np.nan)

    rows: list[dict[str, object]] = []
    for cluster_id, group in work.groupby("_cluster", sort=True):
        means = group[feature_cols].mean()
        z = ((means - overall_mean) / overall_std).fillna(0.0)
        entry: dict[str, object] = {
            "cluster": int(cluster_id),
            "n": int(len(group)),
            "share": round(float(len(group) / len(work)), 6) if len(work) else 0.0,
            "fail_rate_in_cluster": round(
                float(pd.to_numeric(df.loc[group.index, label], errors="coerce").mean()), 6
            )
            if label in df.columns
            else None,
        }
        for col in feature_cols:
            entry[f"mean_{col}"] = round(float(means[col]), 4)
            entry[f"z_{col}"] = round(float(z[col]), 4)
        # 最主要的偏离特征（|z| 最大），便于一句话描述这个簇
        if len(feature_cols):
            top = z.abs().sort_values(ascending=False)
            entry["top_deviating_feature"] = str(top.index[0])
            entry["top_deviation_z"] = round(float(z[top.index[0]]), 4)
        rows.append(entry)
    return pd.DataFrame(rows)


def defect_composition(
    df: pd.DataFrame,
    labels: np.ndarray,
    defect_col: str = "defect_type",
) -> pd.DataFrame:
    """每个簇的缺陷类型构成（行=簇，列=缺陷类型，值=该簇内占比）。

    这是聚类结果与业务语言的接口：簇画像说"工艺条件如何"，
    缺陷构成说"坏成什么样"，两者对得上，簇才有可解释的物理含义。
    """
    if defect_col not in df.columns:
        return pd.DataFrame()
    work = pd.DataFrame(
        {"_cluster": labels, defect_col: df[defect_col].astype("object").to_numpy()}
    )
    table = pd.crosstab(work["_cluster"], work[defect_col], normalize="index")
    table.index.name = "cluster"
    return table


def top_defect_per_cluster(composition: pd.DataFrame) -> dict[int, dict[str, object]]:
    """从构成表里取每个簇占比最高的缺陷类型。"""
    out: dict[int, dict[str, object]] = {}
    if composition.empty:
        return out
    for cluster_id, row in composition.iterrows():
        name = str(row.idxmax())
        out[int(cluster_id)] = {
            "defect_type": name,
            "share": round(float(row.max()), 6),
        }
    return out


def run_clustering(
    df: pd.DataFrame,
    label: str = cfg.LABEL_COL,
    seed: int = cfg.SEED,
    k_candidates: tuple[int, ...] = K_CANDIDATES,
) -> tuple[dict[str, object], dict[str, object]]:
    """对不合格样品做聚类，返回 (可 JSON 化的结果, 绘图用的内存对象)。

    用途写在 `purpose` 字段里：把不合格拆成若干失效模式，分别处置。
    """
    result: dict[str, object] = {
        "purpose": "把「不合格」按工艺条件拆成若干失效模式，便于分别处置（而不是一刀切纠偏）",
        "feature_space": list(cfg.MODEL_NUMERIC),
        "k_candidates": list(k_candidates),
        "n_samples": 0,
        "k_selected": None,
        "silhouette": None,
        "silhouette_by_k": {},
        "inertia_by_k": {},
        "cluster_sizes": {},
        "cluster_share": {},
        "profile": [],
        "defect_composition": {},
        "note": "",
    }
    extras: dict[str, object] = {
        "labels": None,
        "X": None,
        "silhouette_table": pd.DataFrame(),
        "profile": pd.DataFrame(),
        "composition": pd.DataFrame(),
        "pca": None,
    }

    if label not in df.columns:
        result["note"] = f"缺少标签列 {label}，无法筛选不合格样品"
        return result, extras

    fails = df.loc[pd.to_numeric(df[label], errors="coerce") == 1]
    result["n_samples"] = int(len(fails))
    if len(fails) < 4:
        result["note"] = (
            f"不合格样本只有 {len(fails)} 个，无法做有意义的聚类（至少需要 4 个）"
        )
        return result, extras

    X, feature_cols = prepare_cluster_matrix(fails)
    # 全部特征都没有方差时，聚类没有意义（所有点重合）
    valid_cols = [c for c in feature_cols if X[c].std(ddof=0) > 0 and X[c].notna().sum() >= 2]
    if not valid_cols:
        result["note"] = "所有聚类特征都没有方差或没有有效值，聚类无定义"
        return result, extras
    X = X[valid_cols]
    result["feature_space"] = valid_cols

    sil = select_k(X, k_candidates, seed=seed)
    extras["silhouette_table"] = sil
    if sil.empty:
        result["note"] = (
            f"样本量 {len(X)} 不足以支撑任何候选 k（轮廓系数要求 2<=k<=n-1），聚类跳过"
        )
        return result, extras

    result["silhouette_by_k"] = {
        str(int(r["k"])): round(float(r["silhouette"]), 6) for _, r in sil.iterrows()
    }
    result["inertia_by_k"] = {
        str(int(r["k"])): round(float(r["inertia"]), 6) for _, r in sil.iterrows()
    }

    best = sil.loc[sil["silhouette"].idxmax()]
    k = int(best["k"])
    result["k_selected"] = k
    result["silhouette"] = round(float(best["silhouette"]), 6)
    result["k_selection_rule"] = "取轮廓系数最大的 k（同分时取较小的 k）"
    result["calinski_harabasz"] = round(float(best["calinski_harabasz"]), 6)
    result["davies_bouldin"] = round(float(best["davies_bouldin"]), 6)

    labels, model = fit_clusters(X, k, seed=seed)
    profile = cluster_profile(fails, labels, valid_cols, label=label)
    composition = defect_composition(fails, labels)
    tops = top_defect_per_cluster(composition)

    # 把每个簇的主要缺陷类型写进画像，便于一句话读出一个簇
    if tops:
        profile["top_defect_type"] = profile["cluster"].map(
            {cid: v["defect_type"] for cid, v in tops.items()}
        )
        profile["top_defect_share"] = profile["cluster"].map(
            {cid: v["share"] for cid, v in tops.items()}
        )

    result["cluster_sizes"] = {
        str(int(r["cluster"])): int(r["n"]) for _, r in profile.iterrows()
    }
    result["cluster_share"] = {
        str(int(r["cluster"])): float(r["share"]) for _, r in profile.iterrows()
    }
    result["profile"] = profile.to_dict(orient="records")
    result["defect_composition"] = {
        str(int(cid)): {str(k2): round(float(v2), 6) for k2, v2 in row.items()}
        for cid, row in composition.iterrows()
    }
    result["top_defect_per_cluster"] = {
        str(cid): v for cid, v in tops.items()
    }
    result["note"] = (
        f"在不合格样本（n={len(X)}）上对 {len(valid_cols)} 个工艺/原料特征做 KMeans，"
        f"k={k} 由轮廓系数选出（{result['silhouette']}）"
    )

    extras["labels"] = labels
    extras["X"] = X
    extras["profile"] = profile
    extras["composition"] = composition
    extras["samples"] = fails
    # 2D PCA 仅用于把高维簇画出来看一眼，不参与任何统计结论
    if len(valid_cols) >= 2 and len(X) >= 3:
        Xs = model.named_steps["scaler"].transform(
            model.named_steps["imputer"].transform(X)
        )
        pca = PCA(n_components=2, random_state=seed)
        coords = pca.fit_transform(Xs)
        extras["pca"] = {
            "coords": coords,
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        }
    return result, extras
