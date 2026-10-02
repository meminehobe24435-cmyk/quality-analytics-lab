"""统计检验与多重比较校正的测试。

重点覆盖：BH 校正的正确性与单调性、各类检验的退化输入（全同值、单一类别、
样本量不足）、效应量的符号约定、以及簇置换检验的「第一类错误率」（size）是否合理。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qalab import stats as st


# --------------------------------------------------------------------------
# Benjamini-Hochberg
# --------------------------------------------------------------------------
def test_bh_matches_definition_by_brute_force():
    pvals = np.array([0.001, 0.008, 0.039, 0.041, 0.06, 0.074, 0.205, 0.34])
    q, rej = st.benjamini_hochberg(pvals, alpha=0.05)
    m = len(pvals)
    order = np.argsort(pvals)
    # 按定义：q_(i) = min_{j>=i} p_(j) * m / j
    expected_sorted = [
        min(pvals[order[j]] * m / (j + 1) for j in range(i, m)) for i in range(m)
    ]
    got_sorted = q[order]
    assert np.allclose(got_sorted, np.minimum(expected_sorted, 1.0))
    assert rej.tolist() == (q <= 0.05).tolist()


def test_bh_q_values_are_monotone_in_p():
    rng = np.random.default_rng(0)
    pvals = rng.uniform(0, 1, 50)
    q, _ = st.benjamini_hochberg(pvals)
    order = np.argsort(pvals)
    assert np.all(np.diff(q[order]) >= -1e-12), "q 值随 p 值必须单调不减"


def test_bh_all_equal_pvalues_gives_same_q():
    pvals = np.full(10, 0.02)
    q, rej = st.benjamini_hochberg(pvals, alpha=0.05)
    assert np.allclose(q, 0.02)
    assert rej.all()


def test_bh_clips_to_one():
    pvals = np.array([0.9, 0.95, 0.99])
    q, rej = st.benjamini_hochberg(pvals, alpha=0.05)
    assert np.all(q <= 1.0)
    assert not rej.any()


def test_bh_handles_nan_and_empty():
    q, rej = st.benjamini_hochberg([0.01, np.nan, 0.5], alpha=0.05)
    assert np.isnan(q[1])
    assert rej[1] is np.False_ or bool(rej[1]) is False
    assert len(q) == 3

    q2, rej2 = st.benjamini_hochberg([], alpha=0.05)
    assert len(q2) == 0 and len(rej2) == 0

    q3, rej3 = st.benjamini_hochberg([np.nan, np.nan])
    assert np.isnan(q3).all()
    assert not rej3.any()


def test_bh_is_less_conservative_than_bonferroni():
    pvals = np.array([0.001, 0.011, 0.012, 0.013, 0.02])
    q, rej = st.benjamini_hochberg(pvals, alpha=0.05)
    bonferroni = pvals <= 0.05 / len(pvals)
    assert rej.sum() >= bonferroni.sum()


# --------------------------------------------------------------------------
# 效应量
# --------------------------------------------------------------------------
def test_rank_biserial_sign_convention():
    """正号必须表示「第一组取值更大」（开发中曾因符号反了把主因子排到第 8 位）。"""
    a = np.array([10.0, 11.0, 12.0, 13.0])
    b = np.array([1.0, 2.0, 3.0, 4.0])
    res = st.mann_whitney(a, b)
    assert res["p_value"] is not None
    assert res["effect_size"] > 0.9, "第一组明显更大，秩双列相关应为正且接近 1"

    flipped = st.mann_whitney(b, a)
    assert flipped["effect_size"] == pytest.approx(-res["effect_size"])


def test_rank_biserial_extremes():
    assert st.rank_biserial_from_u(16.0, 4, 4) == pytest.approx(1.0)   # 完全分离
    assert st.rank_biserial_from_u(0.0, 4, 4) == pytest.approx(-1.0)
    assert st.rank_biserial_from_u(8.0, 4, 4) == pytest.approx(0.0)    # 无差异
    assert st.rank_biserial_from_u(1.0, 0, 4) is None


def test_cohens_d_direction_and_degenerate():
    a = np.array([5.0, 6.0, 7.0, 8.0])
    b = np.array([1.0, 2.0, 3.0, 4.0])
    # 均值差 4.0；两组合并标准差 = sqrt(5/3)；因此 d = 4 / sqrt(5/3)
    expected = 4.0 / np.sqrt(5.0 / 3.0)
    assert st.cohens_d(a, b) == pytest.approx(expected)
    assert st.cohens_d(a, b) > 0
    assert st.cohens_d(b, a) == pytest.approx(-expected)
    # 无方差 -> None
    assert st.cohens_d([3.0, 3.0, 3.0], [3.0, 3.0, 3.0]) is None
    # 样本量不足 -> None
    assert st.cohens_d([1.0], [2.0, 3.0]) is None


def test_cramers_v_bounds_and_degenerate():
    assert st.cramers_v(0.0, 100, 2, 2) == pytest.approx(0.0)
    assert st.cramers_v(100.0, 100, 2, 2) == pytest.approx(1.0)
    assert st.cramers_v(10.0, 0, 2, 2) is None       # n=0
    assert st.cramers_v(10.0, 100, 1, 3) is None     # 只有 1 行
    assert st.cramers_v(10.0, 100, 3, 1) is None     # 只有 1 列


# --------------------------------------------------------------------------
# 各检验的退化输入
# --------------------------------------------------------------------------
def test_mann_whitney_all_identical_values_is_undefined():
    res = st.mann_whitney([5.0, 5.0, 5.0], [5.0, 5.0])
    assert res["p_value"] is None
    assert "全部取值相同" in res["note"]


def test_mann_whitney_insufficient_samples():
    res = st.mann_whitney([], [1.0, 2.0])
    assert res["p_value"] is None
    assert res["note"] != ""


def test_welch_ttest_reports_effect_and_ci():
    rng = np.random.default_rng(3)
    a = rng.normal(10, 1, 200)
    b = rng.normal(9, 1, 200)
    res = st.welch_ttest(a, b)
    assert res["p_value"] < 0.01
    assert res["mean_diff"] == pytest.approx(a.mean() - b.mean())
    lo, hi = res["ci95"]
    assert lo < res["mean_diff"] < hi
    assert res["effect_metric"] == "cohens_d"
    assert res["effect_size"] > 0


def test_welch_ttest_degenerate_cases():
    assert st.welch_ttest([5.0, 5.0, 5.0], [5.0, 5.0, 5.0])["p_value"] is None
    assert st.welch_ttest([1.0], [2.0, 3.0])["p_value"] is None
    assert st.mean_diff_ci([1.0], [2.0]) is None


def test_chi_square_single_category_is_undefined():
    table = pd.crosstab(pd.Series(["A", "A", "A"]), pd.Series([0, 1, 0]))
    res = st.chi_square_test(table)
    assert res["p_value"] is None
    assert "单一类别" in res["note"] or "维度 < 2" in res["note"]


def test_chi_square_detects_association():
    table = np.array([[80, 20], [20, 80]])
    res = st.chi_square_test(table)
    assert res["p_value"] < 1e-10
    assert res["effect_size"] > 0.4
    assert res["dof"] == 1


def test_chi_square_zero_and_empty_tables():
    res = st.chi_square_test(np.zeros((2, 2)))
    assert res["p_value"] is None
    assert res["n"] == 0
    res2 = st.chi_square_test(np.array([]))
    assert res2["p_value"] is None


def test_point_biserial_matches_pearson():
    rng = np.random.default_rng(5)
    x = rng.normal(size=300)
    y = (x + rng.normal(0, 0.5, 300) > 0).astype(int)
    res = st.point_biserial(y, x)
    assert res["p_value"] is not None
    assert res["effect_size"] == pytest.approx(np.corrcoef(x, y)[0, 1])
    assert res["effect_size"] > 0


def test_point_biserial_degenerate():
    assert st.point_biserial([1, 1, 1], [1.0, 2.0, 3.0])["p_value"] is None  # 单一类别
    assert st.point_biserial([0, 1], [1.0, 2.0])["p_value"] is None          # 样本量 < 3
    assert st.point_biserial([0, 1, 0], [2.0, 2.0, 2.0])["p_value"] is None  # 无方差


def test_spearman_detects_monotone_relation_and_degenerate():
    x = np.arange(50, dtype=float)
    y = x**3
    res = st.spearman_corr(x, y)
    assert res["statistic"] == pytest.approx(1.0)
    assert res["effect_size"] == pytest.approx(1.0)
    assert st.spearman_corr([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])["p_value"] is None
    assert st.spearman_corr([1.0], [2.0])["p_value"] is None


def test_spearman_misses_non_monotone_v_shape():
    """V 形（非单调）关系：秩相关抓不到，这是开发中选择口径时的真实教训。"""
    x = np.linspace(-1, 1, 201)
    y = np.abs(x)
    res = st.spearman_corr(x, y)
    assert abs(res["statistic"]) < 0.15


def test_kruskal_wallis_detects_group_difference():
    rng = np.random.default_rng(11)
    groups = [rng.normal(0, 1, 100), rng.normal(0, 1, 100), rng.normal(3, 1, 100)]
    res = st.kruskal_wallis(groups, ["a", "b", "c"])
    assert res["p_value"] < 1e-10
    assert res["effect_size"] > 0.1
    assert res["n_groups"] == 3


def test_kruskal_wallis_degenerate():
    assert st.kruskal_wallis([[1.0, 2.0]])["p_value"] is None
    assert "组数 < 2" in st.kruskal_wallis([[1.0, 2.0]])["note"]
    assert st.kruskal_wallis([[1.0, 1.0], [1.0, 1.0]])["p_value"] is None


# --------------------------------------------------------------------------
# 簇置换检验
# --------------------------------------------------------------------------
def _clustered_null(n_clusters=60, per_cluster=25, seed=0):
    """构造「因子与标签无关、但簇内相关」的数据。"""
    rng = np.random.default_rng(seed)
    run_ids, x_values, y_values = [], [], []
    for c in range(n_clusters):
        z = rng.normal(0, 1)  # 簇级潜在风险（与因子无关）
        x = rng.normal(0, 1)  # 因子（簇内常数）
        p = 1 / (1 + np.exp(-(-2.0 + z)))
        for u in range(per_cluster):
            run_ids.append(f"R{c:03d}")
            x_values.append(x)
            y_values.append(int(rng.random() < p))
    return np.array(y_values), np.array(x_values), np.array(run_ids)


def test_cluster_permutation_has_reasonable_size():
    """零假设下（因子与标签无关）第一类错误率应接近名义水平 0.05。

    朴素单元级检验在这种数据上会严重膨胀，这正是簇置换存在的理由。
    """
    rejections = 0
    naive_rejections = 0
    n_trials = 40
    for i in range(n_trials):
        y, x, clusters = _clustered_null(seed=i)
        res = st.cluster_permutation_test(
            y, x, clusters, n_perm=200, seed=i, kind="numeric"
        )
        if res["p_value"] < 0.05:
            rejections += 1
        naive = st.mann_whitney(x[y == 1], x[y == 0])
        if naive["p_value"] is not None and naive["p_value"] < 0.05:
            naive_rejections += 1
    assert rejections <= 6, f"簇置换的第一类错误率过高: {rejections}/{n_trials}"
    assert naive_rejections >= rejections, "朴素检验不应比簇置换更保守"


def test_cluster_permutation_detects_real_cluster_level_effect():
    n_clusters, per_cluster = 80, 20
    rng = np.random.default_rng(42)
    run_ids, xs, ys = [], [], []
    for c in range(n_clusters):
        x = rng.normal(0, 1)
        p = 1 / (1 + np.exp(-(-1.5 + 1.8 * x)))
        for _ in range(per_cluster):
            run_ids.append(f"R{c:03d}")
            xs.append(x)
            ys.append(int(rng.random() < p))
    res = st.cluster_permutation_test(
        np.array(ys), np.array(xs), np.array(run_ids), n_perm=300, seed=1, kind="numeric"
    )
    assert res["p_value"] < 0.01
    assert res["statistic"] > 0
    assert res["n_clusters"] == n_clusters


def test_cluster_permutation_requires_cluster_constant_factor():
    y = np.array([0, 1, 0, 1, 1, 0, 0, 1])
    x = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6])  # 簇内变化
    clusters = np.array(["R1", "R1", "R2", "R2", "R3", "R3", "R4", "R4"])
    res = st.cluster_permutation_test(y, x, clusters, n_perm=50, seed=0)
    assert res["p_value"] is None
    assert "簇内不是常数" in res["note"]


def test_cluster_permutation_categorical_kind():
    rng = np.random.default_rng(2)
    run_ids, cats, ys = [], [], []
    for c in range(60):
        cat = "BAD" if c % 4 == 0 else "OK"
        p = 0.35 if cat == "BAD" else 0.05
        for _ in range(20):
            run_ids.append(f"R{c:03d}")
            cats.append(cat)
            ys.append(int(rng.random() < p))
    res = st.cluster_permutation_test(
        np.array(ys), np.array(cats), np.array(run_ids), n_perm=300, seed=3, kind="categorical"
    )
    assert res["p_value"] < 0.01
    assert res["effect_metric"] == "cramers_v"


def test_cluster_permutation_degenerate_inputs():
    y = np.array([1, 1, 1, 1, 1, 1])
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    clusters = np.array(["R1", "R1", "R2", "R2", "R3", "R3"])
    res = st.cluster_permutation_test(y, x, clusters, n_perm=20, seed=0)
    assert res["p_value"] is None  # 单标签

    y2 = np.array([0, 1, 0, 1, 1])
    res2 = st.cluster_permutation_test(y2, x[:5], np.array(["R1", "R1", "R2", "R2", "R3"]), n_perm=20)
    assert res2["p_value"] is None
    assert "簇数量 < 4" in res2["note"]


def test_cluster_permutation_is_reproducible():
    y, x, clusters = _clustered_null(seed=9)
    a = st.cluster_permutation_test(y, x, clusters, n_perm=100, seed=7)
    b = st.cluster_permutation_test(y, x, clusters, n_perm=100, seed=7)
    assert a["p_value"] == b["p_value"]
    assert a["statistic"] == b["statistic"]
    c = st.cluster_permutation_test(y, x, clusters, n_perm=100, seed=8)
    assert c["statistic"] == a["statistic"]  # 统计量与种子无关


def test_cluster_permutation_pvalue_floor():
    """置换次数决定 p 值下界：1/(n_perm+1)，不可能为 0。"""
    n_clusters, per_cluster = 40, 15
    rng = np.random.default_rng(4)
    run_ids, xs, ys = [], [], []
    for c in range(n_clusters):
        x = float(c % 2)  # 完美分离的因子
        p = 0.95 if x > 0.5 else 0.02
        for _ in range(per_cluster):
            run_ids.append(f"R{c:03d}")
            xs.append(x)
            ys.append(int(rng.random() < p))
    res = st.cluster_permutation_test(
        np.array(ys), np.array(xs), np.array(run_ids), n_perm=99, seed=5
    )
    assert res["p_value"] >= 1 / (99 + 1)
    assert res["p_value"] == pytest.approx(1 / 100)
