"""统计检验与多重比较校正。

每个函数都显式处理**退化输入**（全同值、单一类别、样本量不足、全 NaN），
返回 `p_value=None` 并在 `note` 里说明原因，而不是抛异常或返回误导性的 0/1。
这些退化分支都有对应的单元测试。
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats as sps

__all__ = [
    "benjamini_hochberg",
    "cohens_d",
    "cramers_v",
    "chi_square_test",
    "mann_whitney",
    "welch_ttest",
    "point_biserial",
    "spearman_corr",
    "kruskal_wallis",
    "mean_diff_ci",
    "rank_biserial_from_u",
    "cluster_permutation_test",
]


def _clean_pair(a, b) -> tuple[np.ndarray, np.ndarray]:
    """转成有限值 float 数组，去掉 NaN/inf。"""
    x = pd.to_numeric(pd.Series(a), errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(pd.Series(b), errors="coerce").to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    return x, y


# --------------------------------------------------------------------------
# 多重比较校正
# --------------------------------------------------------------------------
def benjamini_hochberg(
    pvalues, alpha: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR 校正。

    返回 (q_values, rejected)。规则：先把 p 升序排，第 i 名的 q 原始值为
    p_i * m / i，再从大到小取累计最小值（保证 q 单调不减），最后截断到 1。
    NaN 的 p 值不参与校正，其 q 与 rejected 均为 NaN / False。
    """
    p = np.asarray(pvalues, dtype=float)
    q = np.full(p.shape, np.nan, dtype=float)
    rejected = np.zeros(p.shape, dtype=bool)
    valid = np.isfinite(p)
    m = int(valid.sum())
    if m == 0:
        return q, rejected

    pv = p[valid]
    order = np.argsort(pv, kind="stable")
    ranked = pv[order]
    ranks = np.arange(1, m + 1, dtype=float)
    q_sorted = ranked * m / ranks
    # 从后往前累计最小，保证 q 值单调不减（否则会有 q_i < q_{i-1} 的矛盾结果）
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)

    q_valid = np.empty(m, dtype=float)
    q_valid[order] = q_sorted
    q[valid] = q_valid
    rejected[valid] = q_valid <= alpha
    return q, rejected


# --------------------------------------------------------------------------
# 效应量
# --------------------------------------------------------------------------
def cohens_d(a, b) -> float | None:
    """Cohen's d（合并标准差口径）。任一组样本量 < 2 或合并方差为 0 时返回 None。"""
    x, y = _clean_pair(a, b)
    if len(x) < 2 or len(y) < 2:
        return None
    n1, n2 = len(x), len(y)
    s2 = ((n1 - 1) * x.var(ddof=1) + (n2 - 1) * y.var(ddof=1)) / (n1 + n2 - 2)
    if s2 <= 0:
        return None
    return float((x.mean() - y.mean()) / math.sqrt(s2))


def rank_biserial_from_u(u: float, n1: int, n2: int) -> float | None:
    """由 Mann-Whitney U 换算秩双列相关（等价于 Cliff's delta）。

    符号约定（显式写死，避免踩坑）：
        U1 = scipy 以第一组为 x 计算的 U 统计量，满足 AUC = U1 / (n1*n2)，
        其中 AUC = P(x_1 > x_2) + 0.5*P(x_1 == x_2)。
        本函数返回 2*AUC - 1，因此 **正号表示第一组取值更大**。

    ⚠️ 开发中实测踩坑：若写成 1 - 2U/(n1*n2)，符号会整体反转，
    曾导致「炉温升高 -> 不合格率升高」被报成负效应，进而把主因子排到第 8 位。
    n1 或 n2 为 0 时返回 None。
    """
    if n1 <= 0 or n2 <= 0:
        return None
    return float(2.0 * u / (n1 * n2) - 1.0)


def cramers_v(chi2: float, n: int, n_rows: int, n_cols: int) -> float | None:
    """Cramér's V。表格退化（少于 2 行或 2 列、n=0）时返回 None。"""
    if n <= 0 or n_rows < 2 or n_cols < 2:
        return None
    denom = n * (min(n_rows, n_cols) - 1)
    if denom <= 0:
        return None
    return float(math.sqrt(chi2 / denom))


# --------------------------------------------------------------------------
# 假设检验
# --------------------------------------------------------------------------
def mann_whitney(a, b, label_a: str = "group_a", label_b: str = "group_b") -> dict:
    """Mann-Whitney U（秩和）检验 + 秩双列相关效应量。"""
    x, y = _clean_pair(a, b)
    out: dict[str, object] = {
        "test": "mann_whitney_u",
        "n_a": int(len(x)),
        "n_b": int(len(y)),
        "group_a": label_a,
        "group_b": label_b,
        "statistic": None,
        "p_value": None,
        "effect_size": None,
        "effect_metric": "rank_biserial",
        "note": "",
    }
    if len(x) < 1 or len(y) < 1:
        out["note"] = "两组中至少一组没有可用样本"
        return out
    combined = np.concatenate([x, y])
    if np.all(combined == combined[0]):
        # scipy 在两组取值完全相同（无秩可排）时会抛 ValueError，这里提前返回
        out["note"] = "全部取值相同，检验无定义"
        return out
    try:
        res = sps.mannwhitneyu(x, y, alternative="two-sided")
    except ValueError as exc:  # 退化为全同值时会抛错
        out["note"] = f"检验无定义: {exc}"
        return out
    out["statistic"] = float(res.statistic)
    out["p_value"] = float(res.pvalue)
    out["effect_size"] = rank_biserial_from_u(float(res.statistic), len(x), len(y))
    out["median_a"] = float(np.median(x))
    out["median_b"] = float(np.median(y))
    return out


def welch_ttest(a, b, label_a: str = "group_a", label_b: str = "group_b") -> dict:
    """Welch t 检验（不假定方差齐）+ 均值差的 95% 置信区间。"""
    x, y = _clean_pair(a, b)
    out: dict[str, object] = {
        "test": "welch_t_test",
        "n_a": int(len(x)),
        "n_b": int(len(y)),
        "group_a": label_a,
        "group_b": label_b,
        "statistic": None,
        "p_value": None,
        "effect_size": cohens_d(x, y),
        "effect_metric": "cohens_d",
        "df": None,
        "mean_a": None,
        "mean_b": None,
        "mean_diff": None,
        "ci95": None,
        "note": "",
    }
    if len(x) < 2 or len(y) < 2:
        out["note"] = "样本量 < 2，t 检验无定义"
        return out
    if x.var(ddof=1) == 0 and y.var(ddof=1) == 0:
        out["note"] = "两组均无方差，t 检验无定义"
        return out
    res = sps.ttest_ind(x, y, equal_var=False)
    out["statistic"] = float(res.statistic)
    out["p_value"] = float(res.pvalue)
    out["mean_a"] = float(x.mean())
    out["mean_b"] = float(y.mean())
    out["mean_diff"] = float(x.mean() - y.mean())
    out["df"] = float(res.df)
    ci = mean_diff_ci(x, y)
    out["ci95"] = [ci[0], ci[1]] if ci else None
    return out


def mean_diff_ci(a, b, alpha: float = 0.05) -> tuple[float, float] | None:
    """Welch 口径的均值差置信区间。"""
    x, y = _clean_pair(a, b)
    if len(x) < 2 or len(y) < 2:
        return None
    se2 = x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y)
    if se2 <= 0:
        return None
    se = math.sqrt(se2)
    # Welch-Satterthwaite 自由度
    num = se2**2
    den = (x.var(ddof=1) / len(x)) ** 2 / (len(x) - 1) + (y.var(ddof=1) / len(y)) ** 2 / (
        len(y) - 1
    )
    if den <= 0:
        return None
    df = num / den
    crit = float(sps.t.ppf(1 - alpha / 2, df))
    diff = float(x.mean() - y.mean())
    return (diff - crit * se, diff + crit * se)


def chi_square_test(table, row_labels=None, col_labels=None) -> dict:
    """卡方独立性检验 + Cramér's V。

    `table` 可以是 2D 计数数组或 pandas 交叉表。任一维度 < 2、总数为 0、
    或存在全零行列时返回 p_value=None（卡方在前两种情况下无定义）。
    """
    arr = np.asarray(table, dtype=float)
    out: dict[str, object] = {
        "test": "chi_square",
        "statistic": None,
        "p_value": None,
        "dof": None,
        "effect_size": None,
        "effect_metric": "cramers_v",
        "n": int(np.nansum(arr)) if arr.size else 0,
        "note": "",
    }
    if arr.size == 0 or arr.ndim != 2:
        out["note"] = "交叉表为空"
        return out
    arr = np.nan_to_num(arr, nan=0.0)
    keep_r = arr.sum(axis=1) > 0
    keep_c = arr.sum(axis=0) > 0
    arr = arr[keep_r][:, keep_c]
    if arr.shape[0] < 2 or arr.shape[1] < 2:
        out["note"] = "剔除全零行列后维度 < 2，卡方检验无定义（可能是单一类别）"
        return out
    res = sps.chi2_contingency(arr, correction=False)
    out["statistic"] = float(res.statistic)
    out["p_value"] = float(res.pvalue)
    out["dof"] = int(res.dof)
    out["n"] = int(arr.sum())
    out["effect_size"] = cramers_v(float(res.statistic), int(arr.sum()), *arr.shape)
    return out


def point_biserial(binary, numeric) -> dict:
    """点双列相关：二值标签与数值因子的线性相关（等价于把标签当 0/1 做 Pearson）。"""
    y = pd.to_numeric(pd.Series(binary), errors="coerce").to_numpy(dtype=float)
    x = pd.to_numeric(pd.Series(numeric), errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    out: dict[str, object] = {
        "test": "point_biserial",
        "statistic": None,
        "p_value": None,
        "effect_size": None,
        "effect_metric": "point_biserial_r",
        "n": int(len(x)),
        "note": "",
    }
    if len(x) < 3 or len(np.unique(y)) < 2:
        out["note"] = "样本量 < 3 或标签只有单一取值，相关无定义"
        return out
    if np.std(x) == 0:
        out["note"] = "因子无方差，相关无定义"
        return out
    res = sps.pointbiserialr(y, x)
    out["statistic"] = float(res.statistic)
    out["p_value"] = float(res.pvalue)
    out["effect_size"] = float(res.statistic)
    return out


def spearman_corr(a, b) -> dict:
    """Spearman 秩相关（用于炉次级：因子 vs 不合格率的单调关系）。"""
    x, y = _clean_pair(a, b)
    out: dict[str, object] = {
        "test": "spearman",
        "statistic": None,
        "p_value": None,
        "effect_size": None,
        "effect_metric": "spearman_rho",
        "n": int(len(x)),
        "note": "",
    }
    if len(x) < 3:
        out["note"] = "样本量 < 3，秩相关无定义"
        return out
    if np.std(x) == 0 or np.std(y) == 0:
        out["note"] = "存在无方差的变量，秩相关无定义"
        return out
    res = sps.spearmanr(x, y)
    out["statistic"] = float(res.statistic)
    out["p_value"] = float(res.pvalue)
    out["effect_size"] = float(res.statistic)
    return out


def kruskal_wallis(groups: list, group_names: list | None = None) -> dict:
    """Kruskal-Wallis 多组秩检验 + epsilon² 效应量（用于炉次级类别因子）。"""
    cleaned: list[np.ndarray] = []
    names: list[str] = []
    for i, g in enumerate(groups):
        arr = pd.to_numeric(pd.Series(g), errors="coerce").to_numpy(dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) >= 1:
            cleaned.append(arr)
            names.append(str(group_names[i]) if group_names is not None else f"g{i}")
    out: dict[str, object] = {
        "test": "kruskal_wallis",
        "statistic": None,
        "p_value": None,
        "effect_size": None,
        "effect_metric": "epsilon_squared",
        "n": int(sum(len(c) for c in cleaned)),
        "n_groups": len(cleaned),
        "note": "",
    }
    if len(cleaned) < 2:
        out["note"] = "有效组数 < 2，检验无定义（可能是单一类别）"
        return out
    stacked = np.concatenate(cleaned)
    if np.std(stacked) == 0:
        out["note"] = "全部取值相同，检验无定义"
        return out
    res = sps.kruskal(*cleaned)
    k = len(cleaned)
    n = len(stacked)
    h = float(res.statistic)
    out["statistic"] = h
    out["p_value"] = float(res.pvalue)
    if n > k:
        out["effect_size"] = float(max(0.0, (h - k + 1) / (n - k)))
    return out


def cluster_permutation_test(
    y,
    x,
    clusters,
    n_perm: int = 1000,
    seed: int = 0,
    kind: str = "numeric",
    label: str = "factor",
) -> dict:
    """按**簇（炉次）置换**的检验，用于修正组内相关（伪重复）导致的 p 值偏小。

    问题背景：同一炉次的单元共享同一个工艺参数与潜在风险，单元之间并不独立。
    直接对 3361 个单元做 Mann-Whitney / 卡方，等于把「96 个独立观测」当成
    「3361 个独立观测」来用，p 值会被系统性高估显著性（anti-conservative）。

    做法：把因子值在**簇之间**随机重排（簇内保持不变），重算统计量，
    得到经验零分布，再算经验 p 值。这样保留了簇内的相关结构。

    要求：因子在簇内必须是常数（工艺参数天然满足）。若不满足，返回
    `p_value=None` 并在 note 里说明，而不是给出一个错误的 p 值。

    statistic 口径：
      numeric      -> 不合格组均值 - 合格组均值
      categorical  -> Cramér's V（因子与标签的关联强度）
    """
    rng = np.random.default_rng(seed)
    y_arr = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(dtype=float)
    cl = pd.Series(clusters).astype("object")
    cl_codes, cl_uniques = pd.factorize(cl)
    n = len(y_arr)
    out: dict[str, object] = {
        "test": f"cluster_permutation_{kind}",
        "factor": label,
        "statistic": None,
        "p_value": None,
        "effect_size": None,
        "effect_metric": "mean_diff_fail_minus_pass" if kind == "numeric" else "cramers_v",
        "n": int(n),
        "n_clusters": int(len(cl_uniques)),
        "n_perm": int(n_perm),
        "note": "",
    }
    if n == 0 or len(cl_uniques) < 4:
        out["note"] = "簇数量 < 4，置换分布不可靠"
        return out

    mask_y = np.isfinite(y_arr)
    if kind == "numeric":
        x_arr = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=float)
        mask = mask_y & np.isfinite(x_arr) & (cl_codes >= 0)
        y_v, x_v, c_v = y_arr[mask], x_arr[mask], cl_codes[mask]
        if len(np.unique(y_v)) < 2:
            out["note"] = "标签只有单一取值，检验无定义"
            return out
        # 检查簇内常量性
        per_cluster_nunique = pd.Series(x_v).groupby(pd.Series(c_v)).nunique()
        if int(per_cluster_nunique.max()) > 1:
            out["note"] = "因子在簇内不是常数，簇置换不适用"
            return out
        n_cl = len(cl_uniques)
        x_cluster = np.full(n_cl, np.nan)
        for code, value in zip(c_v, x_v):
            x_cluster[code] = value
        x_cluster = np.nan_to_num(x_cluster, nan=float(np.nanmedian(x_v)))
        pos, neg = y_v == 1, y_v == 0
        if pos.sum() == 0 or neg.sum() == 0:
            out["note"] = "某一类别没有样本"
            return out

        def stat(xp: np.ndarray) -> float:
            return float(xp[pos].mean() - xp[neg].mean())

        observed = stat(x_cluster[c_v])
        null = np.empty(n_perm, dtype=float)
        for i in range(n_perm):
            perm = rng.permutation(n_cl)
            null[i] = stat(x_cluster[perm][c_v])
        out["statistic"] = observed
        out["effect_size"] = observed
    elif kind == "categorical":
        codes, _ = pd.factorize(pd.Series(x).astype("object"))
        mask = mask_y & (codes >= 0) & (cl_codes >= 0)
        y_v, x_v, c_v = y_arr[mask], codes[mask], cl_codes[mask]
        if len(np.unique(y_v)) < 2 or len(np.unique(x_v)) < 2:
            out["note"] = "标签或因子只有单一取值，检验无定义"
            return out
        per_cluster_nunique = pd.Series(x_v).groupby(pd.Series(c_v)).nunique()
        if int(per_cluster_nunique.max()) > 1:
            out["note"] = "因子在簇内不是常数，簇置换不适用"
            return out
        n_cl = len(cl_uniques)
        n_cat = int(x_v.max()) + 1
        x_cluster = np.zeros(n_cl, dtype=int)
        for code, value in zip(c_v, x_v):
            x_cluster[code] = value

        def stat(xp: np.ndarray) -> float:
            table = np.zeros((n_cat, 2), dtype=float)
            np.add.at(table, (xp, y_v.astype(int)), 1.0)
            keep_r = table.sum(axis=1) > 0
            t = table[keep_r]
            if t.shape[0] < 2:
                return 0.0
            chi2 = float(sps.chi2_contingency(t, correction=False).statistic)
            v = cramers_v(chi2, int(t.sum()), *t.shape)
            return float(v) if v is not None else 0.0

        observed = stat(x_cluster[c_v])
        null = np.empty(n_perm, dtype=float)
        for i in range(n_perm):
            perm = rng.permutation(n_cl)
            null[i] = stat(x_cluster[perm][c_v])
        out["statistic"] = observed
        out["effect_size"] = observed
    else:
        out["note"] = f"未知的 kind: {kind}"
        return out

    # +1 校正：使 p 值不会为 0（置换检验的标准做法）
    n_extreme = int(np.sum(np.abs(null) >= abs(observed) - 1e-12))
    out["p_value"] = float((1 + n_extreme) / (1 + n_perm))
    out["note"] = "簇（炉次）置换检验，已考虑组内相关"
    return out

