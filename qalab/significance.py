"""显著因子识别（单因子筛选 + 多重比较校正）。

对「是否不合格」标签做单因子筛选，用多种口径交叉验证，并且显式区分三个分析层级。
**核心方法论：每个因子要在「它自己变化的层级」上检验，否则就是伪重复。**

【unit_naive】单元级朴素检验（业界最常见的做法，但**在本数据上是无效的**）
  数值因子：Mann-Whitney U 秩检验；类别因子：卡方独立性检验。
  问题：同一炉次内的单元共享同一个工艺参数和同一个潜在风险，观测并不独立。
  把约 5000 个单元当成 5000 个独立观测，等于把有效样本量放大几十倍
  （伪重复 pseudo-replication），p 值会被系统性低估 —— 实测中连「真值零效应」
  的诱饵因子都会被判为显著。

【unit_cluster】单元级 + 簇置换检验（修正伪重复）
  把因子值在**它自己的变化层级**上随机重排（层内互不重复），重算统计量得到
  经验零分布。生产参数在炉次间重排（144 个簇），原料因子在批次间重排（24 个簇）。

【factor_level】因子自己变化层级的汇总口径 —— **本报告的权威口径**
  生产参数按炉次汇总（144 个独立观测），原料因子按批次汇总（24 个独立观测），
  然后：数值因子用 Spearman 秩相关（因子 vs 不合格率），
  类别因子用 Kruskal-Wallis 秩检验 + epsilon²。

【多重比较校正】对所有因子的主口径 p 值做 Benjamini-Hochberg：
被检验的假设集合是「这批因子」，而不是「这批因子 × 每种检验方法」。
补充口径（点双列相关、互信息）只作交叉印证，不进入校正族。

【互信息】sklearn 的 mutual_info_classif，覆盖非线性/非单调关联，
作为「线性与秩口径都没抓到、但确实存在关联」的补充排查手段。

⚠️ 口径与效应形状要匹配（开发中实测）：炉温效应最初设计成 V 形（偏离最优值即
增险），而炉次级主口径是 Spearman 秩相关 —— 非单调效应会被秩相关漏掉，
头号因子在 5 个种子里只检出 2 个。改成单调递增后 5/5 稳定检出。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from . import config as cfg
from . import stats as st

__all__ = [
    "screen_factors",
    "rank_factors",
    "significant_factors",
    "encode_for_mutual_info",
    "mutual_information_scores",
]

ROLE_PRIMARY = "primary"
ROLE_SUPPLEMENTARY = "supplementary"

LEVEL_NAIVE = "unit_naive"
LEVEL_CLUSTER = "unit_cluster"
LEVEL_FACTOR = "factor_level"

MATERIAL_FACTORS: frozenset[str] = frozenset(cfg.MATERIAL_NUMERIC) | {"material_supplier"}
"""只在**批次级**变化的因子：有效样本量 = 批次数（24）。"""


def cluster_key_for_factor(factor: str) -> str:
    """因子应该在哪个层级上做置换/汇总：原料因子按批次，其余按炉次。"""
    return "batch_id" if factor in MATERIAL_FACTORS else "run_id"


def _level_detail(factor: str) -> str:
    return "batch" if factor in MATERIAL_FACTORS else "run"


def _factor_kind(col: str, df: pd.DataFrame | None = None) -> str:
    """判断因子是数值还是类别。

    优先看配置里的列分组（这是权威来源）；对于配置里没有的列
    （例如单元测试里手工构造的因子），退化为按 dtype 判断：
    字符串/object/Categorical 视为类别，其余视为数值。
    """
    if col in cfg.MODEL_CATEGORICAL:
        return "categorical"
    if df is not None and col in df.columns:
        dtype = df[col].dtype
        if dtype == object or isinstance(dtype, pd.CategoricalDtype) or str(dtype).startswith("str"):
            return "categorical"
    return "numeric"




def encode_for_mutual_info(
    df: pd.DataFrame, features: list[str]
) -> tuple[pd.DataFrame, list[bool]]:
    """把特征编码成互信息可用的数值矩阵。

    类别列用**整数编码 + discrete_features=True**，而不是 one-hot：
    one-hot 会把一个类别因子拆成多个特征，互信息被分散到各列，难以归因到因子本身。
    返回 (编码矩阵, discrete 掩码)。
    """
    encoded = pd.DataFrame(index=df.index)
    discrete: list[bool] = []
    for col in features:
        if col not in df.columns:
            continue
        if _factor_kind(col, df) == "categorical":
            codes, _ = pd.factorize(df[col].astype("object"))
            encoded[col] = codes
            discrete.append(True)
        else:
            values = pd.to_numeric(df[col], errors="coerce")
            encoded[col] = values.fillna(values.median())
            discrete.append(False)
    return encoded, discrete


def mutual_information_scores(
    df: pd.DataFrame,
    features: list[str],
    label: str = cfg.LABEL_COL,
    seed: int = cfg.SEED,
) -> dict[str, float | None]:
    """互信息（非线性口径）。固定 random_state，保证可复现。"""
    if label not in df.columns or df.empty:
        return {f: None for f in features}
    X, discrete = encode_for_mutual_info(df, features)
    if X.shape[1] == 0:
        return {}
    y = pd.to_numeric(df[label], errors="coerce")
    mask = y.notna()
    X, y = X.loc[mask], y.loc[mask].astype(int)
    if y.nunique() < 2 or len(y) < 3:
        return {c: None for c in X.columns}
    scores = mutual_info_classif(
        X.to_numpy(dtype=float),
        y.to_numpy(),
        discrete_features=np.array(discrete, dtype=bool),
        random_state=seed,
    )
    return {col: float(s) for col, s in zip(X.columns, scores)}


def _naive_unit_tests(
    df: pd.DataFrame, factors: list[str], label: str
) -> list[dict[str, object]]:
    """单元级朴素检验（忽略组内相关，保留用于对照）。"""
    rows: list[dict[str, object]] = []
    pos, neg = df[label] == 1, df[label] == 0
    for col in factors:
        kind = _factor_kind(col, df)
        base = {
            "level": LEVEL_NAIVE,
            "factor": col,
            "kind": kind,
            "method": "",
            "role": ROLE_PRIMARY,
            "statistic": None,
            "p_value": None,
            "effect_size": None,
            "effect_metric": None,
            "n_used": 0,
            "note": "",
            "fail_median": None,
            "pass_median": None,
        }
        if kind == "numeric":
            mwu = st.mann_whitney(
                pd.to_numeric(df.loc[pos, col], errors="coerce"),
                pd.to_numeric(df.loc[neg, col], errors="coerce"),
                label_a="fail",
                label_b="pass",
            )
            row = dict(base)
            row.update(
                method=mwu["test"],
                statistic=mwu["statistic"],
                p_value=mwu["p_value"],
                effect_size=mwu["effect_size"],
                effect_metric=mwu["effect_metric"],
                n_used=int(mwu["n_a"] + mwu["n_b"]),
                note=mwu["note"],
                fail_median=mwu.get("median_a"),
                pass_median=mwu.get("median_b"),
            )
        else:
            sub = df[[col, label]].dropna()
            chi = st.chi_square_test(pd.crosstab(sub[col].astype("object"), sub[label]))
            row = dict(base)
            row.update(
                method=chi["test"],
                statistic=chi["statistic"],
                p_value=chi["p_value"],
                effect_size=chi["effect_size"],
                effect_metric=chi["effect_metric"],
                n_used=int(chi["n"]),
                note=chi["note"],
                n_levels=int(sub[col].nunique()),
            )
        rows.append(row)
    return rows


def _supplementary_tests(
    df: pd.DataFrame, factors: list[str], label: str
) -> list[dict[str, object]]:
    """补充口径：数值因子的点双列相关（与 MWU 互为印证，但不进入校正族）。"""
    rows: list[dict[str, object]] = []
    for col in factors:
        if _factor_kind(col, df) != "numeric":
            continue
        pb = st.point_biserial(df[label], df[col])
        rows.append(
            {
                "level": LEVEL_NAIVE,
                "factor": col,
                "kind": "numeric",
                "method": pb["test"],
                "role": ROLE_SUPPLEMENTARY,
                "statistic": pb["statistic"],
                "p_value": pb["p_value"],
                "effect_size": pb["effect_size"],
                "effect_metric": pb["effect_metric"],
                "n_used": int(pb["n"]),
                "note": pb["note"],
                "fail_median": None,
                "pass_median": None,
            }
        )
    return rows


def _cluster_robust_tests(
    df: pd.DataFrame,
    factors: list[str],
    label: str,
    n_perm: int,
    seed: int,
) -> list[dict[str, object]]:
    """簇置换检验：把因子值在**它自己的变化层级**上重排，修正伪重复。

    描述性效应量沿用朴素口径的效应量（秩双列相关 / Cramér's V）：
    效应量描述的是「观测到的关联有多强」，与 p 值是否有效是两个问题；
    无效的是 p 值，因为它用了错误的零分布。
    """
    rows: list[dict[str, object]] = []
    for i, col in enumerate(factors):
        kind = _factor_kind(col, df)
        cluster_col = cluster_key_for_factor(col)
        if cluster_col not in df.columns:
            continue
        perm = st.cluster_permutation_test(
            df[label],
            df[col],
            df[cluster_col],
            n_perm=n_perm,
            seed=seed + i,
            kind=kind,
            label=col,
        )
        # 描述性效应量：数值用秩双列相关，类别用 Cramér's V
        if kind == "numeric":
            mwu = st.mann_whitney(
                pd.to_numeric(df.loc[df[label] == 1, col], errors="coerce"),
                pd.to_numeric(df.loc[df[label] == 0, col], errors="coerce"),
            )
            effect, metric = mwu["effect_size"], mwu["effect_metric"]
        else:
            sub = df[[col, label]].dropna()
            chi = st.chi_square_test(pd.crosstab(sub[col].astype("object"), sub[label]))
            effect, metric = chi["effect_size"], chi["effect_metric"]
        rows.append(
            {
                "level": LEVEL_CLUSTER,
                "level_detail": _level_detail(col),
                "level_key": cluster_col,
                "cluster_key": cluster_col,
                "factor": col,
                "kind": kind,
                "method": perm["test"],
                "role": ROLE_PRIMARY,
                "statistic": perm["statistic"],
                "p_value": perm["p_value"],
                "effect_size": effect,
                "effect_metric": metric,
                "n_used": int(perm["n"]),
                "n_clusters": int(perm["n_clusters"]),
                "n_perm": int(perm["n_perm"]),
                "note": perm["note"],
                "fail_median": None,
                "pass_median": None,
            }
        )
    return rows


def _factor_level_tests(
    run_df: pd.DataFrame,
    batch_df: pd.DataFrame,
    factors: list[str],
    outcome: str = "fail_rate",
) -> list[dict[str, object]]:
    """因子自身变化层级的汇总口径：生产参数用炉次级表，原料因子用批次级表。

    数值因子用 Spearman 秩相关；类别因子用 Kruskal-Wallis（对不合格率）。
    """
    rows: list[dict[str, object]] = []
    for col in factors:
        detail = _level_detail(col)
        table = batch_df if detail == "batch" else run_df
        kind = _factor_kind(col, table if table is not None else run_df)
        if table is None or outcome not in getattr(table, "columns", []):
            # 不静默跳过：产出一行「无法检验」，让因子在报告里可见而不是凭空消失
            rows.append(
                {
                    "level": LEVEL_FACTOR,
                    "level_detail": detail,
                    "level_key": "batch_id" if detail == "batch" else "run_id",
                    "cluster_key": "batch_id" if detail == "batch" else "run_id",
                    "factor": col,
                    "kind": kind,
                    "method": "unavailable",
                    "role": ROLE_PRIMARY,
                    "statistic": None,
                    "p_value": None,
                    "effect_size": None,
                    "effect_metric": None,
                    "n_used": 0,
                    "note": f"缺少 {outcome} 列（{detail} 级汇总表未构建），无法检验",
                    "fail_median": None,
                    "pass_median": None,
                }
            )
            continue
        base = {
            "level": LEVEL_FACTOR,
            "level_detail": detail,
            "level_key": "batch_id" if detail == "batch" else "run_id",
            "cluster_key": "batch_id" if detail == "batch" else "run_id",
            "factor": col,
            "kind": kind,
            "method": "",
            "role": ROLE_PRIMARY,
            "statistic": None,
            "p_value": None,
            "effect_size": None,
            "effect_metric": None,
            "n_used": 0,
            "note": "",
            "fail_median": None,
            "pass_median": None,
        }
        if kind == "numeric":
            res = st.spearman_corr(table[col], table[outcome])
            base.update(
                method=res["test"] + "_vs_fail_rate",
                statistic=res["statistic"],
                p_value=res["p_value"],
                effect_size=res["effect_size"],
                effect_metric=res["effect_metric"],
                n_used=int(res["n"]),
                note=f"{detail} 级汇总（{res['n']} 个独立观测）",
            )
        else:
            groups, names = [], []
            for name, g in table.groupby(table[col].astype("object"), dropna=True):
                groups.append(g[outcome])
                names.append(name)
            res = st.kruskal_wallis(groups, names)
            base.update(
                method=res["test"] + "_on_fail_rate",
                statistic=res["statistic"],
                p_value=res["p_value"],
                effect_size=res["effect_size"],
                effect_metric=res["effect_metric"],
                n_used=int(res["n"]),
                note=f"{detail} 级汇总（{res['n']} 个独立观测）",
                n_levels=res.get("n_groups"),
            )
        rows.append(base)
    return rows


def screen_factors(
    unit_df: pd.DataFrame,
    run_df: pd.DataFrame | None = None,
    batch_df: pd.DataFrame | None = None,
    factors: list[str] | None = None,
    label: str = cfg.LABEL_COL,
    alpha: float = cfg.ALPHA,
    seed: int = cfg.SEED,
    n_perm: int = cfg.N_PERM,
) -> pd.DataFrame:
    """跑全部口径的因子筛选，返回长表（一行 = 一个 层级×因子×方法）。"""
    if factors is None:
        factors = [c for c in cfg.MODEL_FEATURES if c in unit_df.columns]

    rows = _naive_unit_tests(unit_df, factors, label)
    rows += _supplementary_tests(unit_df, factors, label)
    rows += _cluster_robust_tests(unit_df, factors, label, n_perm, seed)
    if run_df is not None or batch_df is not None:
        rows += _factor_level_tests(
            run_df if run_df is not None else pd.DataFrame(),
            batch_df if batch_df is not None else pd.DataFrame(),
            factors,
        )

    tests = pd.DataFrame(rows)
    if tests.empty:
        return tests

    # BH 校正：对所有因子的「主口径 p 值族」校正。
    # 注意 unit_cluster 与 factor_level 是两种不同的有效口径，各自成一族分别校正。
    tests["q_value"] = np.nan
    tests["significant"] = False
    for level in tests["level"].unique():
        mask = (tests["level"] == level) & (tests["role"] == ROLE_PRIMARY)
        q, rej = st.benjamini_hochberg(
            tests.loc[mask, "p_value"].to_numpy(dtype=float), alpha=alpha
        )
        tests.loc[mask, "q_value"] = q
        tests.loc[mask, "significant"] = rej

    mi = mutual_information_scores(unit_df, factors, label, seed)
    tests["mutual_info"] = tests["factor"].map(mi)
    return tests



def rank_factors(tests: pd.DataFrame, alpha: float = cfg.ALPHA) -> pd.DataFrame:
    """把长表整理成「一行一个因子」的重要性排序表。

    排序依据是**因子自身变化层级**的 q 值（`factor_q_value`）：
    生产参数用炉次级汇总（144 个独立观测），原料因子用批次级汇总（24 个独立观测），
    都是有效口径里统计效率最高的。同 q 值按效应量绝对值降序。

    `verdict` 综合三种口径：
      稳健显著      -> 因子层级口径与单元级簇置换都显著
      因子层级显著  -> 只有因子层级口径显著（单元级二值标签效率较低）
      簇置换显著    -> 只有单元级簇置换显著
      伪重复假阳性  -> 朴素单元级显著、但两种有效口径都不显著
      不显著        -> 三种口径都不显著
    """
    columns = [
        "rank",
        "factor",
        "kind",
        "factor_level",
        "n_independent",
        "factor_method",
        "factor_statistic",
        "factor_p_value",
        "factor_q_value",
        "factor_effect_size",
        "factor_significant",
        "unit_method",
        "unit_statistic",
        "unit_p_value",
        "unit_q_value",
        "unit_effect_size",
        "unit_effect_metric",
        "unit_significant",
        "unit_naive_method",
        "unit_naive_statistic",
        "unit_naive_p_value",
        "unit_naive_q_value",
        "unit_naive_significant",
        "unit_supplementary_method",
        "unit_supplementary_p_value",
        "mutual_info",
        "verdict",
    ]
    if tests.empty:
        return pd.DataFrame(columns=columns)

    out_rows = []
    for factor, grp in tests.groupby("factor", sort=False):
        def pick(level: str, role: str = ROLE_PRIMARY):
            sel = grp[(grp["level"] == level) & (grp["role"] == role)]
            return sel.iloc[0] if len(sel) else None

        naive = pick(LEVEL_NAIVE)
        cluster = pick(LEVEL_CLUSTER)
        own = pick(LEVEL_FACTOR)
        supp = pick(LEVEL_NAIVE, ROLE_SUPPLEMENTARY)

        naive_sig = bool(naive["significant"]) if naive is not None else False
        cluster_sig = bool(cluster["significant"]) if cluster is not None else False
        own_sig = bool(own["significant"]) if own is not None else False

        if own_sig and cluster_sig:
            verdict = "稳健显著（因子层级 + 簇置换一致）"
        elif own_sig:
            verdict = "因子层级显著（单元级簇置换未达显著）"
        elif cluster_sig:
            verdict = "仅单元级簇置换显著（因子层级未达显著）"
        elif naive_sig:
            verdict = "仅朴素单元级显著（有效口径下消失，判定为伪重复假阳性）"
        else:
            verdict = "不显著"

        out_rows.append(
            {
                "factor": factor,
                "kind": grp["kind"].iloc[0],
                "factor_level": own["level_detail"] if own is not None else None,
                "n_independent": int(own["n_used"]) if own is not None else None,
                "factor_method": own["method"] if own is not None else None,
                "factor_statistic": own["statistic"] if own is not None else None,
                "factor_p_value": own["p_value"] if own is not None else None,
                "factor_q_value": own["q_value"] if own is not None else None,
                "factor_effect_size": own["effect_size"] if own is not None else None,
                "factor_significant": own_sig,
                "unit_method": cluster["method"] if cluster is not None else None,
                "unit_statistic": cluster["statistic"] if cluster is not None else None,
                "unit_p_value": cluster["p_value"] if cluster is not None else None,
                "unit_q_value": cluster["q_value"] if cluster is not None else None,
                "unit_effect_size": cluster["effect_size"] if cluster is not None else None,
                "unit_effect_metric": cluster["effect_metric"] if cluster is not None else None,
                "unit_significant": cluster_sig,
                "unit_naive_method": naive["method"] if naive is not None else None,
                "unit_naive_statistic": naive["statistic"] if naive is not None else None,
                "unit_naive_p_value": naive["p_value"] if naive is not None else None,
                "unit_naive_q_value": naive["q_value"] if naive is not None else None,
                "unit_naive_significant": naive_sig,
                "unit_supplementary_method": supp["method"] if supp is not None else None,
                "unit_supplementary_p_value": supp["p_value"] if supp is not None else None,
                "mutual_info": grp["mutual_info"].iloc[0],
                "verdict": verdict,
            }
        )

    ranking = pd.DataFrame(out_rows)
    ranking["_abs_effect"] = pd.to_numeric(
        ranking["unit_effect_size"], errors="coerce"
    ).abs().fillna(0.0)
    ranking = ranking.sort_values(
        ["factor_q_value", "_abs_effect", "factor"],
        ascending=[True, False, True],
        na_position="last",
    ).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    return ranking.drop(columns=["_abs_effect"])


def significant_factors(ranking: pd.DataFrame, level: str = "factor") -> list[str]:
    """按指定层级取出显著因子名单。

    `level` 取值对应排序列名前缀：`factor`（因子自身层级，默认）、
    `unit`（单元级簇置换）、`unit_naive`（未修正伪重复的朴素检验）。
    """
    col = f"{level}_significant"
    if ranking.empty or col not in ranking.columns:
        return []
    return ranking.loc[ranking[col], "factor"].tolist()


