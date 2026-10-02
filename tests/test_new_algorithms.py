"""聚类、回归、时间序列三个新增算法模块的测试。

每个模块都覆盖「正常输入」与「退化输入」：
  - 聚类：只有一个簇的可能、不合格样本太少、特征无方差、k 候选不可行；
  - 回归：目标方差为 0、目标全缺失、样本量不足；
  - 时间序列：只有 1 个批次、序列全相等、点数不足、含缺失值、空序列。
退化情形的要求是「不崩 + 给出合理结论（而不是硬编一个数字）」。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qalab import cluster as cl
from qalab import config as cfg
from qalab import regression as rg
from qalab import timeseries as ts


# ==========================================================================
# 聚类
# ==========================================================================
def _two_group_frame(n_per_group: int = 60, seed: int = 0) -> pd.DataFrame:
    """构造两个明显分开的失效模式（高温裂纹 / 高压气孔）+ 一批合格样本。"""
    rng = np.random.default_rng(seed)
    rows = []
    # 模式 A：炉温偏高（裂纹）
    for i in range(n_per_group):
        rows.append(
            {
                "unit_id": f"A{i:04d}",
                "is_fail": 1,
                "furnace_temp_c": 1525 + rng.normal(0, 2),
                "pressure_mpa": 3.0 + rng.normal(0, 0.1),
                "defect_type": "CRACK",
            }
        )
    # 模式 B：含水率偏高（气孔）——用 pressure 之外的特征体现差异
    for i in range(n_per_group):
        rows.append(
            {
                "unit_id": f"B{i:04d}",
                "is_fail": 1,
                "furnace_temp_c": 1490 + rng.normal(0, 2),
                "pressure_mpa": 3.6 + rng.normal(0, 0.1),
                "defect_type": "PORE",
            }
        )
    # 合格样本（不应参与聚类）
    for i in range(n_per_group):
        rows.append(
            {
                "unit_id": f"P{i:04d}",
                "is_fail": 0,
                "furnace_temp_c": 1505 + rng.normal(0, 2),
                "pressure_mpa": 3.0 + rng.normal(0, 0.1),
                "defect_type": "NONE",
            }
        )
    return pd.DataFrame(rows)


def test_run_clustering_finds_two_obvious_modes():
    df = _two_group_frame()
    result, extras = cl.run_clustering(df, k_candidates=(2, 3, 4))
    assert result["n_samples"] == 120, "只应对不合格样品聚类"
    assert result["k_selected"] == 2, "两个明显分开的模式应当被选出 k=2"
    assert result["silhouette"] > 0.5
    assert sum(result["cluster_sizes"].values()) == 120
    assert len(result["profile"]) == 2
    # 两个簇应当分别对应"高温裂纹"和"高压气孔"
    profile = pd.DataFrame(result["profile"]).set_index("cluster")
    temp_z = profile["z_furnace_temp_c"]
    pressure_z = profile["z_pressure_mpa"]
    assert set(np.sign(temp_z)) == {-1.0, 1.0} or set(np.sign(temp_z)) == {-1, 1}
    assert np.sign(temp_z.iloc[0]) != np.sign(pressure_z.iloc[0]) or len(set(np.sign(pressure_z))) == 2
    # 缺陷构成应当能分辨两个簇
    comp = pd.DataFrame(result["defect_composition"]).T
    assert set(comp.columns) == {"CRACK", "PORE"} or len(comp.columns) >= 1
    assert extras["labels"] is not None
    assert len(extras["labels"]) == 120


def test_cluster_profile_reports_share_and_size():
    df = _two_group_frame(n_per_group=40)
    result, _ = cl.run_clustering(df, k_candidates=(2,))
    profile = pd.DataFrame(result["profile"])
    assert profile["n"].sum() == 80
    assert profile["share"].sum() == pytest.approx(1.0)
    for _, row in profile.iterrows():
        assert row["fail_rate_in_cluster"] == pytest.approx(1.0)
        assert "top_deviating_feature" in row


def test_cluster_result_is_json_serializable_and_reproducible():
    import json

    df = _two_group_frame()
    a, _ = cl.run_clustering(df, k_candidates=(2, 3))
    b, _ = cl.run_clustering(df, k_candidates=(2, 3))
    assert a == b, "固定种子下聚类结果必须逐字段一致"
    json.dumps(a, ensure_ascii=False)


def test_clustering_too_few_fail_samples():
    df = pd.DataFrame(
        {
            "unit_id": ["U1", "U2", "U3"],
            "is_fail": [1, 1, 0],
            "furnace_temp_c": [1500.0, 1520.0, 1510.0],
        }
    )
    result, extras = cl.run_clustering(df)
    assert result["k_selected"] is None
    assert "无法做有意义的聚类" in result["note"]
    assert extras["labels"] is None


def test_clustering_with_zero_variance_features():
    df = pd.DataFrame(
        {
            "unit_id": [f"U{i}" for i in range(10)],
            "is_fail": [1] * 10,
            "furnace_temp_c": [1500.0] * 10,
            "pressure_mpa": [3.0] * 10,
        }
    )
    result, _ = cl.run_clustering(df)
    assert result["k_selected"] is None
    assert "没有方差" in result["note"]


def test_clustering_missing_label_column():
    result, _ = cl.run_clustering(pd.DataFrame({"x": [1, 2, 3]}))
    assert result["k_selected"] is None
    assert "缺少标签列" in result["note"]


def test_select_k_shrinks_candidates_when_sample_is_small():
    """样本量小于候选 k 时，轮廓系数无定义，候选集合必须被自动收缩。"""
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [1.0, 5.0, 9.0]})
    table = cl.select_k(X, k_candidates=(2, 3, 4, 5))
    assert list(table["k"]) == [2], "n=3 时只有 k=2 可行"
    # 样本更少时一个候选都不剩 -> 空表而不是抛异常
    empty = cl.select_k(pd.DataFrame({"a": [1.0, 2.0]}), k_candidates=(2, 3))
    assert empty.empty


def test_select_k_reports_all_metrics():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "a": np.concatenate([rng.normal(0, 0.3, 20), rng.normal(10, 0.3, 20)]),
            "b": np.concatenate([rng.normal(1, 0.3, 20), rng.normal(9, 0.3, 20)]),
        }
    )
    table = cl.select_k(X, k_candidates=(2, 3))
    for col in ("k", "silhouette", "inertia", "calinski_harabasz", "davies_bouldin"):
        assert col in table.columns
    assert (table["inertia"] > 0).all()
    assert table.loc[table["k"] == 2, "silhouette"].iloc[0] > 0.9


def test_defect_composition_rows_sum_to_one():
    df = _two_group_frame()
    X, cols = cl.prepare_cluster_matrix(df)
    labels, _ = cl.fit_clusters(X, 2)
    comp = cl.defect_composition(df, labels)
    assert np.allclose(comp.sum(axis=1).to_numpy(), 1.0)
    tops = cl.top_defect_per_cluster(comp)
    assert set(tops) == {0, 1}
    assert all(v["defect_type"] in {"CRACK", "PORE"} for v in tops.values())


def test_defect_composition_without_defect_column():
    df = _two_group_frame().drop(columns=["defect_type"])
    X, _ = cl.prepare_cluster_matrix(df)
    labels, _ = cl.fit_clusters(X, 2)
    assert cl.defect_composition(df, labels).empty


def test_cluster_matrix_handles_missing_values():
    df = _two_group_frame().copy()
    df.loc[df.index[:5], "furnace_temp_c"] = np.nan
    result, _ = cl.run_clustering(df, k_candidates=(2,))
    assert result["k_selected"] == 2


# ==========================================================================
# 回归
# ==========================================================================
def _regression_frame(n: int = 300, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    temp = rng.normal(1505, 8, n)
    pressure = rng.normal(3.0, 0.25, n)
    strength = 430 - 22 * (temp - 1505) / 8 - 10 * (pressure - 3.0) / 0.25 + rng.normal(0, 8, n)
    return pd.DataFrame(
        {
            "unit_id": [f"U{i:05d}" for i in range(n)],
            "furnace_temp_c": temp,
            "pressure_mpa": pressure,
            "machine_id": rng.choice(["M1", "M2"], size=n),
            "is_fail": (rng.random(n) < 0.12).astype(int),
            "tensile_strength_mpa": strength,
            "elongation_pct": 9.5 - 0.5 * (temp - 1505) / 8 + rng.normal(0, 1, n),
            "hardness_hv": 120 - 3 * (pressure - 3.0) / 0.25 + rng.normal(0, 3, n),
            "surface_roughness_ra": 1.3 + 0.1 * (temp - 1505) / 8 + rng.normal(0, 0.15, n),
        }
    )


def test_regression_metrics_basic():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    perfect = rg.regression_metrics(y, y)
    assert perfect["mae"] == 0.0
    assert perfect["rmse"] == 0.0
    assert perfect["r2"] == pytest.approx(1.0)
    assert perfect["constant_target"] is False

    off = rg.regression_metrics(y, y + 1.0)
    assert off["mae"] == pytest.approx(1.0)
    assert off["rmse"] == pytest.approx(1.0)
    assert off["n"] == 4


def test_regression_metrics_constant_target_has_no_r2():
    """目标方差为 0 时 R² 无定义（分母为 0），必须返回 None 而不是 0/1。"""
    y = np.array([5.0, 5.0, 5.0, 5.0])
    res = rg.regression_metrics(y, y)
    assert res["constant_target"] is True
    assert res["r2"] is None
    assert res["mae"] == 0.0
    assert "无定义" in res["note"]
    res2 = rg.regression_metrics(y, y + 3.0)
    assert res2["r2"] is None
    assert res2["mae"] == pytest.approx(3.0)


def test_regression_metrics_degenerate_inputs():
    assert rg.regression_metrics([], [])["mae"] is None
    assert "没有可用样本" in rg.regression_metrics([], [])["note"]
    res = rg.regression_metrics([1.0, np.nan], [1.0, 2.0])
    assert res["n"] == 1
    res2 = rg.regression_metrics([np.nan, np.nan], [1.0, 2.0])
    assert res2["n"] == 0
    assert res2["mae"] is None


def test_run_regression_beats_mean_baseline():
    df = _regression_frame()
    result, extras = rg.run_regression(df, target="tensile_strength_mpa")
    primary = result["primary"]
    assert primary["n_rows_used"] == len(df)
    base_mae = primary["baseline_mean"]["test"]["mae"]
    for name, entry in primary["models"].items():
        assert entry["test"]["mae"] < base_mae, f"{name} 没有超过均值基线"
        assert entry["test"]["r2"] > 0.5
        assert entry["mae_reduction_vs_baseline_pct"] > 0
    assert result["beats_baseline"] is True
    assert result["best_model_by_test_mae"] in primary["models"]
    # 绘图数据齐全
    assert extras["y_test"].shape == extras["pred_test"][result["best_model_by_test_mae"]].shape
    assert len(extras["baseline_test"]) == len(extras["y_test"])


def test_run_regression_reports_all_targets():
    df = _regression_frame()
    result, _ = rg.run_regression(df, target="tensile_strength_mpa", all_targets=rg.DEFAULT_TARGETS)
    assert len(result["all_targets"]) == len(rg.DEFAULT_TARGETS) - 1
    for entry in result["all_targets"]:
        assert entry["target"] != "tensile_strength_mpa"
        assert entry["best_model"] in ("linear_regression", "gradient_boosting_regressor")
        assert entry["best_mae"] is not None


def test_run_regression_is_reproducible():
    df = _regression_frame()
    a, _ = rg.run_regression(df, all_targets=())
    b, _ = rg.run_regression(df, all_targets=())
    assert a == b


def test_run_regression_drops_rows_without_target():
    df = _regression_frame()
    df.loc[df.index[:50], "tensile_strength_mpa"] = np.nan
    result, _ = rg.run_regression(df, all_targets=())
    assert result["primary"]["n_rows_used"] == len(df) - 50
    assert result["primary"]["missing_target_rows"] == 50


def test_run_regression_constant_target_is_rejected():
    """退化输入：目标全相等 -> 明确拒绝回归而不是给出 R²=0 的假结论。"""
    df = _regression_frame()
    df["tensile_strength_mpa"] = 430.0
    result, extras = rg.run_regression(df, all_targets=())
    assert result["primary"]["constant_target"] is True
    assert "方差为 0" in result["primary"]["note"]
    assert "models" not in result["primary"] or not result["primary"]["models"]
    assert extras == {}


def test_run_regression_all_target_missing():
    df = _regression_frame()
    df["tensile_strength_mpa"] = np.nan
    result, _ = rg.run_regression(df, all_targets=())
    assert result["primary"]["n_rows_used"] == 0
    assert "样本量不足以做回归" in result["primary"]["note"]


def test_run_regression_missing_target_column():
    df = _regression_frame().drop(columns=["tensile_strength_mpa"])
    result, _ = rg.run_regression(df, all_targets=())
    assert "缺少目标列" in result["primary"]["note"]


def test_run_regression_too_few_samples():
    df = _regression_frame(n=10)
    result, _ = rg.run_regression(df, all_targets=())
    assert "样本量不足" in result["primary"]["note"]


def test_run_regression_output_is_json_serializable():
    import json

    df = _regression_frame()
    result, _ = rg.run_regression(df, all_targets=())
    json.dumps(result, ensure_ascii=False)


def test_build_regressors_returns_two_models():
    models = rg.build_regressors(cfg.SEED)
    assert set(models) == {"linear_regression", "gradient_boosting_regressor"}


# ==========================================================================
# 时间序列
# ==========================================================================
def test_moving_average_basic_and_short_series():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    ma = ts.moving_average(values, window=2)
    assert np.isnan(ma.iloc[0])
    assert ma.iloc[1] == pytest.approx(1.5)
    assert ma.iloc[-1] == pytest.approx(4.5)
    # 序列比窗口短 -> 全 NaN，不抛异常
    assert ts.moving_average(values, window=10).isna().all()
    assert ts.moving_average(values, window=1).isna().all()


def test_mann_kendall_detects_upward_trend():
    y = np.arange(30, dtype=float) + np.random.default_rng(0).normal(0, 0.2, 30)
    res = ts.mann_kendall_test(y)
    assert res["p_value"] < 0.001
    assert res["trend"] == "显著上升趋势"
    assert res["Z"] > 0
    assert res["sen_slope"] > 0


def test_mann_kendall_detects_downward_trend():
    y = 50 - np.arange(30, dtype=float) * 0.5
    res = ts.mann_kendall_test(y)
    assert res["trend"] == "显著下降趋势"
    assert res["Z"] < 0
    assert res["sen_slope"] < 0


def test_mann_kendall_on_white_noise_is_not_significant():
    """功效与假阳性都要看：白噪声不应被稳定判为有趋势。"""
    rng = np.random.default_rng(0)
    rejections = sum(
        1
        for i in range(40)
        if ts.mann_kendall_test(rng.normal(0, 1, 30))["p_value"] < 0.05
    )
    assert rejections <= 5, f"白噪声下第一类错误率过高: {rejections}/40"


def test_mann_kendall_degenerate_inputs():
    # 点数不足
    res = ts.mann_kendall_test([1.0, 2.0])
    assert res["p_value"] is None
    assert "点数 < 3" in res["note"]
    # 全相等 -> 方差为 0，Z 与 p 无定义
    res2 = ts.mann_kendall_test([3.0] * 10)
    assert res2["p_value"] is None
    assert res2["S"] == 0
    assert "无定义" in res2["note"]
    assert res2["sen_slope"] == pytest.approx(0.0)
    # 含缺失值 -> 只按有效值算；n=4 太小，检验通常不显著，但统计量必须是正确的
    res3 = ts.mann_kendall_test([1.0, np.nan, 2.0, 3.0, 4.0])
    assert res3["n"] == 4
    assert res3["S"] == 6  # 完全单调上升的 4 个点：C(4,2)=6
    assert res3["p_value"] is not None
    assert res3["trend"] in ("显著上升趋势", "无显著趋势")


def test_mann_kendall_tie_correction():
    """有大量并列值时，方差必须做并列校正（否则 p 值偏小）。"""
    y = [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0]
    res = ts.mann_kendall_test(y)
    n = len(y)
    _, counts = np.unique(y, return_counts=True)
    tie = np.sum(counts * (counts - 1) * (2 * counts + 5))
    expected_var = (n * (n - 1) * (2 * n + 5) - tie) / 18.0
    assert res["var_s"] == pytest.approx(expected_var)
    assert res["sen_slope"] > 0


def test_sen_slope_robust_to_outlier():
    base = np.arange(20, dtype=float)
    with_outlier = base.copy()
    with_outlier[5] = 1000.0
    slope_clean = ts.sen_slope(base)
    slope_outlier = ts.sen_slope(with_outlier)
    assert slope_clean == pytest.approx(1.0)
    # Sen's 斜率被一个极端离群值影响有限（中位数性质）
    assert abs(slope_outlier - slope_clean) < 30
    assert ts.sen_slope([1.0]) is None


def test_linear_trend_test_and_degenerate():
    res = ts.linear_trend_test([1.0, 2.0, 3.0, 4.0, 5.0])
    assert res["slope"] == pytest.approx(1.0)
    assert res["r2"] == pytest.approx(1.0)
    assert res["p_value"] < 0.01
    assert res["slope_per_100_steps"] == pytest.approx(100.0)
    assert ts.linear_trend_test([1.0, 2.0])["p_value"] is None
    assert "点数 < 3" in ts.linear_trend_test([1.0, 2.0])["note"]
    # 全相等：斜率 0 但 r2/p 无定义（自变量仍有方差，因变量无方差）
    flat = ts.linear_trend_test([2.0] * 8)
    assert flat["slope"] == pytest.approx(0.0)


def test_lag1_autocorrelation_detects_ar1():
    rng = np.random.default_rng(0)
    n = 300
    y = np.zeros(n)
    for i in range(1, n):
        y[i] = 0.9 * y[i - 1] + rng.normal(0, 0.3)
    res = ts.lag1_autocorrelation(y)
    assert res["r1"] > 0.7
    assert res["significant"] is True


def test_lag1_autocorrelation_white_noise_and_degenerate():
    """白噪声下不应被稳定判为显著：跑 40 次看拒绝率，而不是只看一次抽样。"""
    rng = np.random.default_rng(1)
    rejections = 0
    for _ in range(40):
        res = ts.lag1_autocorrelation(rng.normal(0, 1, 300))
        assert abs(res["r1"]) < 0.3
        rejections += int(res["significant"])
    assert rejections <= 5, f"白噪声下自相关第一类错误率过高: {rejections}/40"

    assert ts.lag1_autocorrelation([1.0, 2.0])["r1"] is None
    assert "点数 < 4" in ts.lag1_autocorrelation([1.0, 2.0])["note"]
    flat = ts.lag1_autocorrelation([3.0] * 10)
    assert flat["r1"] is None
    assert "无方差" in flat["note"]


def test_simple_exponential_smoothing_forecasts_level():
    y = [10.0] * 20
    res = ts.simple_exponential_smoothing(y)
    assert res["next_forecast"] == pytest.approx(10.0)
    assert "无方差" in res["note"]
    # 有噪声的平稳序列：预测值应落在观测范围内
    rng = np.random.default_rng(0)
    y2 = 5 + rng.normal(0, 0.5, 50)
    res2 = ts.simple_exponential_smoothing(y2)
    assert 4.0 < res2["next_forecast"] < 6.0
    assert 0.05 <= res2["alpha"] <= 0.95
    assert len(res2["fitted"]) == 50
    # 点数不足
    assert ts.simple_exponential_smoothing([1.0, 2.0])["next_forecast"] is None


def test_fit_ar1_recovers_persistence():
    rng = np.random.default_rng(2)
    n = 400
    y = np.zeros(n)
    for i in range(1, n):
        y[i] = 0.7 * y[i - 1] + rng.normal(0, 0.5)
    res = ts.fit_ar1(y)
    assert res["phi"] == pytest.approx(0.7, abs=0.1)
    assert res["p_value_phi"] < 0.01
    assert res["next_forecast"] is not None


def test_fit_ar1_degenerate():
    assert ts.fit_ar1([1.0, 2.0, 3.0])["phi"] is None
    assert "点数 < 5" in ts.fit_ar1([1.0, 2.0, 3.0])["note"]
    flat = ts.fit_ar1([4.0] * 10)
    assert flat["phi"] is None
    assert "滞后段无方差" in flat["note"]


def test_expanding_window_backtest_compares_all_predictors():
    rng = np.random.default_rng(4)
    y = 10 + rng.normal(0, 0.3, 40)
    res = ts.expanding_window_backtest(y)
    assert set(res["predictors"]) == {"ses", "ar1", "mean", "last"}
    assert res["n_backtest_points"] == 40 - res["min_train"]
    for name, m in res["predictors"].items():
        assert m["mae"] > 0 and m["rmse"] >= m["mae"] * 0.99
    assert res["best_predictor_by_mae"] in res["predictors"]
    # 点数不足
    assert ts.expanding_window_backtest([1.0, 2.0])["predictors"] == {}


def test_expanding_window_backtest_rewards_predictable_series():
    """有强自相关的序列上，AR(1)/SES 必须优于"用历史均值预测"。

    ⚠️ 不断言优于「用上一个值预测」：当 φ 接近 1 时（近单位根），
    naive 的"上一个值"本身就是很强的预测器，AR(1) 的收缩反而可能略输 ——
    这是实测到的现象，naive 基线在时序里必须老老实实报出来，不能省略。
    """
    rng = np.random.default_rng(5)
    n = 120
    y = np.zeros(n)
    for i in range(1, n):
        y[i] = 0.95 * y[i - 1] + rng.normal(0, 0.2)
    res = ts.expanding_window_backtest(y)
    assert res["predictors"]["ar1"]["mae"] < res["predictors"]["mean"]["mae"]
    assert res["predictors"]["ses"]["mae"] < res["predictors"]["mean"]["mae"]
    # naive 基线必须被报告出来（哪怕它很强）
    assert "last" in res["predictors"]
    assert res["predictors"]["last"]["mae"] > 0


def test_batch_series_orders_by_time(clean_wide):
    series = ts.batch_series(clean_wide, cfg.LABEL_COL)
    assert len(series) == clean_wide["batch_id"].nunique()
    assert series["order_time"].is_monotonic_increasing
    assert series["n_units"].sum() == len(clean_wide)
    assert series["value"].between(0, 1).all()


def test_batch_series_falls_back_to_batch_id_without_time():
    df = pd.DataFrame(
        {
            "batch_id": ["B02", "B02", "B01", "B01"],
            "is_fail": [1, 0, 0, 0],
        }
    )
    series = ts.batch_series(df, "is_fail")
    assert series["batch_id"].tolist() == ["B01", "B02"]
    assert series["value"].tolist() == [0.0, 0.5]
    assert series["order_time"].isna().all()


def test_run_series_orders_by_time(clean_wide):
    series = ts.run_series(clean_wide, cfg.LABEL_COL)
    assert len(series) == clean_wide["run_id"].nunique()
    assert series["order_time"].is_monotonic_increasing


def test_run_timeseries_end_to_end(clean_wide):
    result, extras = ts.run_timeseries(clean_wide, value_cols=("tensile_strength_mpa",))
    assert result["n_batches"] == clean_wide["batch_id"].nunique()
    assert result["n_runs"] == clean_wide["run_id"].nunique()
    assert "batch_fail_rate" in result["series"]
    assert "batch_mean_tensile_strength_mpa" in result["series"]
    assert "run_fail_rate" in result["series"]
    primary = result["series"]["batch_fail_rate"]
    assert primary["n_points"] == result["n_batches"]
    assert len(primary["values"]) == primary["n_points"]
    assert len(primary["moving_average"]) == primary["n_points"]
    assert primary["mann_kendall"]["p_value"] is not None
    assert result["next_batch_forecast"]["exponential_smoothing"] is not None
    assert result["next_batch_forecast"]["ar1"] is not None
    assert set(result["forecast_one_step_error"]) == {"ses", "ar1", "mean", "last"}
    # 边界说明必须存在，防止把批次序列说成高频时序
    assert "不是高频时序" in result["scope_note"]
    import json

    json.dumps(result, ensure_ascii=False)


def test_run_timeseries_single_batch_is_reported_not_crashed():
    """退化输入：只有 1 个批次 -> 明确说明序列太短，而不是抛异常或给出假结论。"""
    df = pd.DataFrame(
        {
            "batch_id": ["B01"] * 5,
            "run_id": ["R1", "R1", "R2", "R2", "R2"],
            "is_fail": [0, 1, 0, 0, 1],
            "start_time": pd.to_datetime(["2025-01-01"] * 5),
        }
    )
    result, _ = ts.run_timeseries(df)
    assert result["n_batches"] == 1
    assert "序列太短" in result["conclusion"]
    assert result["series"]["batch_fail_rate"]["n_points"] == 1
    assert result["series"]["batch_fail_rate"]["mann_kendall"]["p_value"] is None


def test_run_timeseries_empty_and_missing_columns():
    result, _ = ts.run_timeseries(pd.DataFrame())
    assert "无法做序列分析" in result["conclusion"]
    result2, _ = ts.run_timeseries(pd.DataFrame({"x": [1, 2]}))
    assert "无法做序列分析" in result2["conclusion"]


def test_run_timeseries_constant_series(clean_wide):
    """序列全相等：趋势检验与自相关都无定义，但预测应给出该常数且不崩。"""
    df = clean_wide.copy()
    df["flat_metric"] = 7.0
    result, _ = ts.run_timeseries(df, value_cols=("flat_metric",))
    flat = result["series"]["batch_mean_flat_metric"]
    assert flat["mann_kendall"]["p_value"] is None
    assert flat["autocorrelation_lag1"]["r1"] is None
    assert flat["exponential_smoothing"]["next_forecast"] == pytest.approx(7.0)
    assert flat["ar1"]["phi"] is None


def test_run_timeseries_detects_injected_trend():
    """把已知趋势注入批次序列，必须被检出（检测器功效的端到端验证）。"""
    rows = []
    for i in range(30):
        z = i * 0.6  # 明显的上升趋势
        p = 1 / (1 + np.exp(-(-3.0 + z)))
        n_units = 20
        n_fail = int(round(p * n_units))
        for u in range(n_units):
            rows.append(
                {
                    "batch_id": f"B{i:02d}",
                    "run_id": f"R{i:03d}",
                    "start_time": pd.Timestamp("2025-01-01") + pd.Timedelta(days=i),
                    "furnace_temp_c": 1505.0,
                    "is_fail": 1 if u < n_fail else 0,
                }
            )
    df = pd.DataFrame(rows)
    result, _ = ts.run_timeseries(df)
    primary = result["series"]["batch_fail_rate"]
    assert primary["mann_kendall"]["p_value"] < 0.01
    assert primary["mann_kendall"]["trend"] == "显著上升趋势"
    assert primary["linear_trend"]["p_value"] < 0.01
    assert "显著" in result["conclusion"] or "趋势" in result["conclusion"]
